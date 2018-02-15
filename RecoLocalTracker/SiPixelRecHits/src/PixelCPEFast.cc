#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

// this is needed to get errors from templates
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplate.h"
#include "DataFormats/DetId/interface/DetId.h"


// Services
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "EventFilter/SiPixelRawToDigi/plugins/cudaCheck.h"

#include <iostream>

  namespace {

        struct Stat {
          Stat():c(0){}
          ~Stat(){std::cout << "CPE stat " << c << ' ' << maxx << ' ' << maxd << std::endl;}
          std::atomic<uint32_t> c;
          float maxx=0.f, maxd=0.0f;
        };
        Stat statx, staty; 
  }

namespace {
   constexpr float micronsToCm = 1.0e-4;
   const bool MYDEBUG = false;
}

//-----------------------------------------------------------------------------
//!  The constructor.
//-----------------------------------------------------------------------------
PixelCPEFast::PixelCPEFast(edm::ParameterSet const & conf,
                                 const MagneticField * mag,
                                 const TrackerGeometry& geom,
                                 const TrackerTopology& ttopo,
                                 const SiPixelLorentzAngle * lorentzAngle,
                                 const SiPixelGenErrorDBObject * genErrorDBObject,
                                 const SiPixelLorentzAngle * lorentzAngleWidth)
                  : PixelCPEBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, nullptr,lorentzAngleWidth,0) {
   
   EdgeClusterErrorX_ = conf.getParameter<double>("EdgeClusterErrorX");
   EdgeClusterErrorY_ = conf.getParameter<double>("EdgeClusterErrorY");
   
   
   UseErrorsFromTemplates_    = conf.getParameter<bool>("UseErrorsFromTemplates");
   TruncatePixelCharge_       = conf.getParameter<bool>("TruncatePixelCharge");
   
   
   // Use errors from templates or from GenError
   if ( UseErrorsFromTemplates_ ) {
     if ( !SiPixelGenError::pushfile( *genErrorDBObject_, thePixelGenError_) )
            throw cms::Exception("InvalidCalibrationLoaded")
            << "ERROR: GenErrors not filled correctly. Check the sqlite file. Using SiPixelTemplateDBObject version "
            << ( *genErrorDBObject_ ).version();
     if(MYDEBUG) std::cout<<"Loaded genErrorDBObject v"<<( *genErrorDBObject_ ).version()<< std::endl;
   }  else {
     if(MYDEBUG) std::cout<<" Use simple parametrised errors "<< std::endl;
   } // if ( UseErrorsFromTemplates_ )
   
   
   // Rechit errors in case other, more correct, errors are not vailable
   // This are constants. Maybe there is a more efficienct way to store them.
      xerr_barrel_l1_= {0.00115, 0.00120, 0.00088};
      xerr_barrel_l1_def_=0.01030;
      yerr_barrel_l1_= {0.00375,0.00230,0.00250,0.00250,0.00230,0.00230,0.00210,0.00210,0.00240};
      yerr_barrel_l1_def_=0.00210;
      xerr_barrel_ln_= {0.00115, 0.00120, 0.00088};
      xerr_barrel_ln_def_=0.01030;
      yerr_barrel_ln_= {0.00375,0.00230,0.00250,0.00250,0.00230,0.00230,0.00210,0.00210,0.00240};
      yerr_barrel_ln_def_=0.00210;
      xerr_endcap_= {0.0020, 0.0020};
      xerr_endcap_def_=0.0020;
      yerr_endcap_= {0.00210};
      yerr_endcap_def_=0.00075;



   fillParamsForGpu();   
   
}

void PixelCPEFast::fillParamsForGpu() {


  m_commonParamsGPU.theThicknessB = m_DetParams.front().theThickness;
  m_commonParamsGPU.theThicknessE = m_DetParams.back().theThickness;
  m_commonParamsGPU.thePitchX = m_DetParams[0].thePitchX;
  m_commonParamsGPU.thePitchY = m_DetParams[0].thePitchY;

  uint32_t oldLayer = 0;
  m_detParamsGPU.resize(m_DetParams.size());
  for (auto i=0U; i<m_DetParams.size(); ++i) {
    auto & p=m_DetParams[i];
    auto & g=m_detParamsGPU[i];

    assert(p.theDet->index()==int(i));

    assert(m_commonParamsGPU.thePitchY==p.thePitchY);    
    assert(m_commonParamsGPU.thePitchX==p.thePitchX);
    // assert(m_commonParamsGPU.theThickness==p.theThickness);

    g.isBarrel = GeomDetEnumerators::isBarrel(p.thePart);
    g.isPosZ = p.theDet->surface().position().z()>0;
    g.layer = ttopo_.layer(p.theDet->geographicalId());
    g.index=i; // better be!
    g.rawId = p.theDet->geographicalId();
   
    assert( (g.isBarrel ?m_commonParamsGPU.theThicknessB : m_commonParamsGPU.theThicknessE) ==p.theThickness );

    // if (m_commonParamsGPU.theThickness!=p.theThickness)   
    //  std::cout << i << (g.isBarrel ? "B " : "E ") << m_commonParamsGPU.theThickness<<"!="<<p.theThickness << std::endl;

    if (oldLayer != g.layer) {
      oldLayer = g.layer;
      std::cout << "new layer at " << i << (g.isBarrel ? " B  " :  (g.isPosZ ? " E+ " : " E- ")) << g.layer << " starting at " << g.rawId << std::endl;
    }

    g.shiftX = 0.5f*p.lorentzShiftInCmX;
    g.shiftY = 0.5f*p.lorentzShiftInCmY;
    g.chargeWidthX = p.lorentzShiftInCmX * p.widthLAFractionX;
    g.chargeWidthY = p.lorentzShiftInCmY * p.widthLAFractionY;

    g.x0 = p.theOrigin.x();
    g.y0 = p.theOrigin.y();
    g.z0 = p.theOrigin.z();

    auto vv = p.theDet->surface().position();
    auto rr = pixelCPEforGPU::Rotation(p.theDet->surface().rotation());
    g.frame =  pixelCPEforGPU::Frame(vv.x(),vv.y(),vv.z(),rr);

  }

  // and now copy to device...
  cudaCheck(cudaMalloc((void**) & h_paramsOnGPU.m_commonParams, sizeof(pixelCPEforGPU::CommonParams)));
  cudaCheck(cudaMalloc((void**) & h_paramsOnGPU.m_detParams, m_detParamsGPU.size()*sizeof(pixelCPEforGPU::DetParams)));
  cudaCheck(cudaMalloc((void**) & d_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU)));

  cudaCheck(cudaMemcpy(d_paramsOnGPU, &h_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(h_paramsOnGPU.m_commonParams,&m_commonParamsGPU,sizeof(pixelCPEforGPU::CommonParams), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(h_paramsOnGPU.m_detParams, m_detParamsGPU.data(), m_detParamsGPU.size()*sizeof(pixelCPEforGPU::DetParams), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
}

PixelCPEFast::~PixelCPEFast() {

  cudaFree(h_paramsOnGPU.m_commonParams);
  cudaFree(h_paramsOnGPU.m_detParams);
  cudaFree(d_paramsOnGPU);

}



PixelCPEBase::ClusterParam* PixelCPEFast::createClusterParam(const SiPixelCluster & cl) const
{
   return new ClusterParamGeneric(cl);
}



//-----------------------------------------------------------------------------
//! Hit position in the local frame (in cm).  Unlike other CPE's, this
//! one converts everything from the measurement frame (in channel numbers)
//! into the local frame (in centimeters).
//-----------------------------------------------------------------------------
LocalPoint
PixelCPEFast::localPosition(DetParam const & theDetParam, ClusterParam & theClusterParamBase) const
{
   
   ClusterParamGeneric & theClusterParam = static_cast<ClusterParamGeneric &>(theClusterParamBase);

   assert(!theClusterParam.with_track_angle); 
   
   float chargeWidthX = (theDetParam.lorentzShiftInCmX * theDetParam.widthLAFractionX);
   float chargeWidthY = (theDetParam.lorentzShiftInCmY * theDetParam.widthLAFractionY);
   float shiftX = 0.5f*theDetParam.lorentzShiftInCmX;
   float shiftY = 0.5f*theDetParam.lorentzShiftInCmY;
   
   if ( UseErrorsFromTemplates_ ) {
      
      float qclus = theClusterParam.theCluster->charge();
      float locBz = theDetParam.bz;
      float locBx = theDetParam.bx;
      //cout << "PixelCPEFast::localPosition(...) : locBz = " << locBz << endl;
      
      theClusterParam.pixmx  = std::numeric_limits<int>::max();  // max pixel charge for truncation of 2-D cluster

      theClusterParam.sigmay = -999.9; // CPE Generic y-error for multi-pixel cluster
      theClusterParam.sigmax = -999.9; // CPE Generic x-error for multi-pixel cluster
      theClusterParam.sy1    = -999.9; // CPE Generic y-error for single single-pixel
      theClusterParam.sy2    = -999.9; // CPE Generic y-error for single double-pixel cluster
      theClusterParam.sx1    = -999.9; // CPE Generic x-error for single single-pixel cluster
      theClusterParam.sx2    = -999.9; // CPE Generic x-error for single double-pixel cluster
      
      float dummy;
      
      SiPixelGenError gtempl(thePixelGenError_);
      int gtemplID_ = theDetParam.detTemplateId;
      
      theClusterParam.qBin_ = gtempl.qbin( gtemplID_, theClusterParam.cotalpha, theClusterParam.cotbeta, locBz, locBx, qclus, 
                                          false,
                                          theClusterParam.pixmx, theClusterParam.sigmay, dummy,
                                          theClusterParam.sigmax, dummy, theClusterParam.sy1,
                                          dummy, theClusterParam.sy2, dummy, theClusterParam.sx1,
                                          dummy, theClusterParam.sx2, dummy );
      
      
      theClusterParam.sigmax = theClusterParam.sigmax * micronsToCm;
      theClusterParam.sx1 = theClusterParam.sx1 * micronsToCm;
      theClusterParam.sx2 = theClusterParam.sx2 * micronsToCm;
      
      theClusterParam.sigmay = theClusterParam.sigmay * micronsToCm;
      theClusterParam.sy1 = theClusterParam.sy1 * micronsToCm;
      theClusterParam.sy2 = theClusterParam.sy2 * micronsToCm;
      
   } // if ( UseErrorsFromTemplates_ )
   else {
     theClusterParam.qBin_ = 0;
   }
   
   int Q_f_X;        //!< Q of the first  pixel  in X
   int Q_l_X;        //!< Q of the last   pixel  in X
   int Q_f_Y;        //!< Q of the first  pixel  in Y
   int Q_l_Y;        //!< Q of the last   pixel  in Y
   collect_edge_charges( theClusterParam,
                        Q_f_X, Q_l_X,
                        Q_f_Y, Q_l_Y,
                        UseErrorsFromTemplates_ && TruncatePixelCharge_
                        );
   
   //--- Find the inner widths along X and Y in one shot.  We
   //--- compute the upper right corner of the inner pixels
   //--- (== lower left corner of upper right pixel) and
   //--- the lower left corner of the inner pixels
   //--- (== upper right corner of lower left pixel), and then
   //--- subtract these two points in the formula.
   
   //--- Upper Right corner of Lower Left pixel -- in measurement frame
   uint16_t llx = theClusterParam.theCluster->minPixelRow()+1;
   uint16_t lly = theClusterParam.theCluster->minPixelCol()+1;
   
   //--- Lower Left corner of Upper Right pixel -- in measurement frame
   uint16_t urx = theClusterParam.theCluster->maxPixelRow();
   uint16_t ury = theClusterParam.theCluster->maxPixelCol();
   
   auto llxl = phase1PixelTopology::localX(llx);   
   auto	llyl = phase1PixelTopology::localY(lly);
   auto	urxl = phase1PixelTopology::localX(urx);
   auto uryl = phase1PixelTopology::localY(ury);

   
   float xPos =
   generic_position_formula( theClusterParam.theCluster->sizeX(),
                            Q_f_X, Q_l_X,
                            llxl, urxl,
                            chargeWidthX,   // lorentz shift in cm
                            theDetParam.theThickness,
                            theClusterParam.cotalpha,
                            theDetParam.thePitchX,
                            phase1PixelTopology::isBigPixX( theClusterParam.theCluster->minPixelRow() ),
                            phase1PixelTopology::isBigPixX( theClusterParam.theCluster->maxPixelRow() )
                           );   
   
   // apply the lorentz offset correction
   xPos = xPos + shiftX + theDetParam.thePitchX*float(phase1PixelTopology::xOffset);
   
   float yPos =
   generic_position_formula( theClusterParam.theCluster->sizeY(),
                            Q_f_Y, Q_l_Y,
                            llyl, uryl,
                            chargeWidthY,   // lorentz shift in cm
                            theDetParam.theThickness,
                            theClusterParam.cotbeta,
                            theDetParam.thePitchY,
                            phase1PixelTopology::isBigPixY( theClusterParam.theCluster->minPixelCol() ),
                            phase1PixelTopology::isBigPixY( theClusterParam.theCluster->maxPixelCol() )
                           );   
   // apply the lorentz offset correction
   yPos = yPos + shiftY + theDetParam.thePitchY*float(phase1PixelTopology::yOffset);


   {
     // ok now do GPU like ...

     pixelCPEforGPU::ClusParams cp;

     
     cp.minRow[0] = theClusterParam.theCluster->minPixelRow();
     cp.maxRow[0] = theClusterParam.theCluster->maxPixelRow();
     cp.minCol[0] = theClusterParam.theCluster->minPixelCol();
     cp.maxCol[0] = theClusterParam.theCluster->maxPixelCol();

      cp.Q_f_X[0] = Q_f_X;
      cp.Q_l_X[0] = Q_l_X;
      cp.Q_f_Y[0] = Q_f_Y;
      cp.Q_l_Y[0] = Q_l_Y;

      auto ind = theDetParam.theDet->index();
      pixelCPEforGPU::position(m_commonParamsGPU, m_detParamsGPU[ind],cp,0);
      auto xg = cp.xpos[0];     
      auto yg =	cp.ypos[0];

      if(std::abs(xPos-xg)>0.001) {++statx.c; statx.maxx=std::max(statx.maxx,xPos);}
      statx.maxd=std::max(std::abs(xPos-xg), statx.maxd);
      if(std::abs(yPos-yg)>0.001) {++staty.c; staty.maxx=std::max(staty.maxx,yPos);}
      staty.maxd=std::max(std::abs(yPos-yg), staty.maxd);
      if(std::abs(xPos-xg)>0.001 || std::abs(yPos-yg)>0.001) 
          std::cout << (m_detParamsGPU[ind].isBarrel ? "B " : "E ") << xPos <<'/'<<xPos-xg << ' ' << yPos <<'/'<<yPos-yg << ' ' << cp.maxRow[0]-cp.minRow[0] << ',' << cp.maxCol[0]-cp.minCol[0] 
                    << ' ' << urxl - llxl << ',' << uryl - llyl << std::endl;
   
   }

   //--- Now put the two together
   LocalPoint pos_in_local( xPos, yPos );
   return pos_in_local;
}



//-----------------------------------------------------------------------------
//!  A generic version of the position formula.  Since it works for both
//!  X and Y, in the interest of the simplicity of the code, all parameters
//!  are passed by the caller.  The only class variable used by this method
//!  is the theThickness, since that's common for both X and Y.
//-----------------------------------------------------------------------------
float
PixelCPEFast::
generic_position_formula( int size,                //!< Size of this projection.
                         int Q_f,              //!< Charge in the first pixel.
                         int Q_l,              //!< Charge in the last pixel.
                         uint16_t upper_edge_first_pix, //!< As the name says.
                         uint16_t lower_edge_last_pix,  //!< As the name says.
                         float lorentz_shift,   //!< L-shift at half thickness
                         float theThickness,   //detector thickness
                         float cot_angle,        //!< cot of alpha_ or beta_
                         float pitch,            //!< thePitchX or thePitchY
                         bool first_is_big,       //!< true if the first is big
                         bool last_is_big        //!< true if the last is big
                        )
{
   
   float geom_center = 0.5f * pitch*float( upper_edge_first_pix + lower_edge_last_pix );
   
   //--- The case of only one pixel in this projection is separate.  Note that
   //--- here first_pix == last_pix, so the average of the two is still the
   //--- center of the pixel.
   if ( size == 1 ) {return geom_center;}

   float W_eff; // the compiler detects the logic below (and warns if buggy!!!!0 
   bool simple=true;
   if (size==2) {   
     //--- Width of the clusters minus the edge (first and last) pixels.
     //--- In the note, they are denoted x_F and x_L (and y_F and y_L)
     assert(lower_edge_last_pix>=upper_edge_first_pix);
     float W_inner      =  pitch * float(lower_edge_last_pix-upper_edge_first_pix);  // in cm
   
     //--- Predicted charge width from geometry
     float W_pred = theThickness * cot_angle                     // geometric correction (in cm)
                    - lorentz_shift;                    // (in cm) &&& check fpix!
   
     W_eff = std::abs( W_pred ) - W_inner;

     //--- If the observed charge width is inconsistent with the expectations
     //--- based on the track, do *not* use W_pred-W_innner.  Instead, replace
     //--- it with an *average* effective charge width, which is the average
     //--- length of the edge pixels.
     //
     simple = ( W_eff < 0.0f ) | ( W_eff > pitch );  // this produces "large" regressions for very small numeric differences...

   }
   if (simple) {
     //--- Total length of the two edge pixels (first+last)
     float sum_of_edge = 2.0f;
     if (first_is_big) sum_of_edge += 1.0f;
     if (last_is_big)  sum_of_edge += 1.0f;
     W_eff = pitch * 0.5f * sum_of_edge;  // ave. length of edge pixels (first+last) (cm)
   }
   
   
   //--- Finally, compute the position in this projection
   float Qdiff = Q_l - Q_f;
   float Qsum  = Q_l + Q_f;
   
   //--- Temporary fix for clusters with both first and last pixel with charge = 0
   if(Qsum==0) Qsum=1.0f;
   float hit_pos = geom_center + 0.5f*(Qdiff/Qsum) * W_eff;
   
   return hit_pos;
}


//-----------------------------------------------------------------------------
//!  Collect the edge charges in x and y, in a single pass over the pixel vector.
//!  Calculate charge in the first and last pixel projected in x and y
//!  and the inner cluster charge, projected in x and y.
//-----------------------------------------------------------------------------
void
PixelCPEFast::
collect_edge_charges(ClusterParam & theClusterParamBase,  //!< input, the cluster
                     int & Q_f_X,              //!< output, Q first  in X
                     int & Q_l_X,              //!< output, Q last   in X
                     int & Q_f_Y,              //!< output, Q first  in Y
                     int & Q_l_Y,              //!< output, Q last   in Y
   	       	     bool truncate
)
{
   ClusterParamGeneric & theClusterParam = static_cast<ClusterParamGeneric &>(theClusterParamBase);
   
   // Initialize return variables.
   Q_f_X = Q_l_X = 0;
   Q_f_Y = Q_l_Y = 0;
   
   
   // Obtain boundaries in index units
   int xmin = theClusterParam.theCluster->minPixelRow();
   int xmax = theClusterParam.theCluster->maxPixelRow();
   int ymin = theClusterParam.theCluster->minPixelCol();
   int ymax = theClusterParam.theCluster->maxPixelCol();
   
   
   // Iterate over the pixels.
   int isize = theClusterParam.theCluster->size();
   for (int i = 0;  i != isize; ++i)
   {
      auto const & pixel = theClusterParam.theCluster->pixel(i);
      // ggiurgiu@fnal.gov: add pixel charge truncation
      int pix_adc = pixel.adc;
      if ( truncate )
         pix_adc = std::min(pix_adc, theClusterParam.pixmx );
      
      //
      // X projection
      if ( pixel.x == xmin ) Q_f_X += pix_adc;
      if ( pixel.x == xmax ) Q_l_X += pix_adc;
      //
      // Y projection
      if ( pixel.y == ymin ) Q_f_Y += pix_adc;
      if ( pixel.y == ymax ) Q_l_Y += pix_adc;
   }
}


//==============  INFLATED ERROR AND ERRORS FROM DB BELOW  ================

//-------------------------------------------------------------------------
//  Hit error in the local frame
//-------------------------------------------------------------------------
LocalError
PixelCPEFast::localError(DetParam const & theDetParam,  ClusterParam & theClusterParamBase) const
{
   
   ClusterParamGeneric & theClusterParam = static_cast<ClusterParamGeneric &>(theClusterParamBase);
   
   // Default errors are the maximum error used for edge clusters.
   // These are determined by looking at residuals for edge clusters
   float xerr = EdgeClusterErrorX_ * micronsToCm;
   float yerr = EdgeClusterErrorY_ * micronsToCm;
   
   
   // Find if cluster is at the module edge.
   int maxPixelCol = theClusterParam.theCluster->maxPixelCol();
   int maxPixelRow = theClusterParam.theCluster->maxPixelRow();
   int minPixelCol = theClusterParam.theCluster->minPixelCol();
   int minPixelRow = theClusterParam.theCluster->minPixelRow();
   
   bool edgex =  phase1PixelTopology::isEdgeX(minPixelRow) | phase1PixelTopology::isEdgeX(maxPixelRow);
   bool edgey =  phase1PixelTopology::isEdgeY(minPixelCol) | phase1PixelTopology::isEdgeY(maxPixelCol);
   
   unsigned int sizex = theClusterParam.theCluster->sizeX();
   unsigned int sizey = theClusterParam.theCluster->sizeY();
   
   // Find if cluster contains double (big) pixels.
   bool bigInX = theDetParam.theRecTopol->containsBigPixelInX( minPixelRow, maxPixelRow );
   bool bigInY = theDetParam.theRecTopol->containsBigPixelInY( minPixelCol, maxPixelCol );
   
   if (UseErrorsFromTemplates_ ) {
      //
      // Use template errors
      
      if ( !edgex ) { // Only use this for non-edge clusters
         if ( sizex == 1 ) {
            if ( !bigInX ) {xerr = theClusterParam.sx1;}
            else           {xerr = theClusterParam.sx2;}
         } else {xerr = theClusterParam.sigmax;}
      }
      
      if ( !edgey ) { // Only use for non-edge clusters
         if ( sizey == 1 ) {
            if ( !bigInY ) {yerr = theClusterParam.sy1;}
            else           {yerr = theClusterParam.sy2;}
         } else {yerr = theClusterParam.sigmay;}
      }
      
   } else  { // simple errors
      
      // This are the simple errors, hardcoded in the code
      //cout << "Track angles are not known " << endl;
      //cout << "Default angle estimation which assumes track from PV (0,0,0) does not work." << endl;
      
      if ( GeomDetEnumerators::isTrackerPixel(theDetParam.thePart) ) {
         if(GeomDetEnumerators::isBarrel(theDetParam.thePart)) {
            
            DetId id = (theDetParam.theDet->geographicalId());
            int layer=ttopo_.layer(id);
            if ( layer==1 ) {
               if ( !edgex ) {
                  if ( sizex<=xerr_barrel_l1_.size() ) xerr=xerr_barrel_l1_[sizex-1];
                  else xerr=xerr_barrel_l1_def_;
               }
               
               if ( !edgey ) {
                  if ( sizey<=yerr_barrel_l1_.size() ) yerr=yerr_barrel_l1_[sizey-1];
                  else yerr=yerr_barrel_l1_def_;
               }
            } else{  // layer 2,3
               if ( !edgex ) {
                  if ( sizex<=xerr_barrel_ln_.size() ) xerr=xerr_barrel_ln_[sizex-1];
                  else xerr=xerr_barrel_ln_def_;
               }
               
               if ( !edgey ) {
                  if ( sizey<=yerr_barrel_ln_.size() ) yerr=yerr_barrel_ln_[sizey-1];
                  else yerr=yerr_barrel_ln_def_;
               }
            }
            
         } else { // EndCap
            
            if ( !edgex ) {
               if ( sizex<=xerr_endcap_.size() ) xerr=xerr_endcap_[sizex-1];
               else xerr=xerr_endcap_def_;
            }
            
            if ( !edgey ) {
               if ( sizey<=yerr_endcap_.size() ) yerr=yerr_endcap_[sizey-1];
               else yerr=yerr_endcap_def_;
            }
         } // end endcap
      }
      
   } // end 
   
   auto xerr_sq = xerr*xerr; 
   auto yerr_sq = yerr*yerr;
   
   return LocalError( xerr_sq, 0, yerr_sq );
   
}
