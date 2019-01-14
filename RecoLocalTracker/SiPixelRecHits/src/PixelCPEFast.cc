#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"
#include "HeterogeneousCore/CUDAServices/interface/numberOfCUDADevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"

// Services
// this is needed to get errors from templates

namespace {
   constexpr float micronsToCm = 1.0e-4;
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
                                 const SiPixelLorentzAngle * lorentzAngleWidth) :
  PixelCPEBase(conf, mag, geom, ttopo, lorentzAngle, genErrorDBObject, nullptr, lorentzAngleWidth, 0)
{
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
   }
   
   // Rechit errors in case other, more correct, errors are not vailable
   // This are constants. Maybe there is a more efficienct way to store them.
   xerr_barrel_l1_      = { 0.00115, 0.00120, 0.00088 };
   xerr_barrel_l1_def_  = 0.01030;
   yerr_barrel_l1_      = { 0.00375, 0.00230, 0.00250, 0.00250, 0.00230, 0.00230, 0.00210, 0.00210, 0.00240 };
   yerr_barrel_l1_def_  = 0.00210;
   xerr_barrel_ln_      = { 0.00115, 0.00120, 0.00088};
   xerr_barrel_ln_def_  = 0.01030;
   yerr_barrel_ln_      = { 0.00375, 0.00230, 0.00250, 0.00250, 0.00230, 0.00230, 0.00210, 0.00210, 0.00240 };
   yerr_barrel_ln_def_  = 0.00210;
   xerr_endcap_         = { 0.0020, 0.0020 };
   xerr_endcap_def_     = 0.0020;
   yerr_endcap_         = { 0.00210 };
   yerr_endcap_def_     = 0.00075;

   fillParamsForGpu();   
}

const pixelCPEforGPU::ParamsOnGPU *PixelCPEFast::getGPUProductAsync(cuda::stream_t<>& cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cuda::stream_t<>& stream) {
      // and now copy to device...
      cudaCheck(cudaMalloc((void**) & data.h_paramsOnGPU.m_commonParams, sizeof(pixelCPEforGPU::CommonParams)));
      cudaCheck(cudaMalloc((void**) & data.h_paramsOnGPU.m_detParams, this->m_detParamsGPU.size()*sizeof(pixelCPEforGPU::DetParams)));
      cudaCheck(cudaMalloc((void**) & data.d_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU)));

      cudaCheck(cudaMemcpyAsync(data.d_paramsOnGPU, &data.h_paramsOnGPU, sizeof(pixelCPEforGPU::ParamsOnGPU), cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(data.h_paramsOnGPU.m_commonParams, &this->m_commonParamsGPU, sizeof(pixelCPEforGPU::CommonParams), cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(data.h_paramsOnGPU.m_detParams, this->m_detParamsGPU.data(), this->m_detParamsGPU.size()*sizeof(pixelCPEforGPU::DetParams), cudaMemcpyDefault, stream.id()));
    });
  return data.d_paramsOnGPU;
}

void PixelCPEFast::fillParamsForGpu() {
  m_commonParamsGPU.theThicknessB = m_DetParams.front().theThickness;
  m_commonParamsGPU.theThicknessE = m_DetParams.back().theThickness;
  m_commonParamsGPU.thePitchX = m_DetParams[0].thePitchX;
  m_commonParamsGPU.thePitchY = m_DetParams[0].thePitchY;

  //uint32_t oldLayer = 0;
  m_detParamsGPU.resize(m_DetParams.size());
  for (auto i=0U; i<m_DetParams.size(); ++i) {
    auto & p=m_DetParams[i];
    auto & g=m_detParamsGPU[i];

    assert(p.theDet->index()==int(i));
    assert(m_commonParamsGPU.thePitchY==p.thePitchY);    
    assert(m_commonParamsGPU.thePitchX==p.thePitchX);
    //assert(m_commonParamsGPU.theThickness==p.theThickness);

    g.isBarrel = GeomDetEnumerators::isBarrel(p.thePart);
    g.isPosZ = p.theDet->surface().position().z()>0;
    g.layer = ttopo_.layer(p.theDet->geographicalId());
    g.index=i; // better be!
    g.rawId = p.theDet->geographicalId();
   
    assert( (g.isBarrel ?m_commonParamsGPU.theThicknessB : m_commonParamsGPU.theThicknessE) ==p.theThickness );

    //if (m_commonParamsGPU.theThickness!=p.theThickness)   
    //  std::cout << i << (g.isBarrel ? "B " : "E ") << m_commonParamsGPU.theThickness<<"!="<<p.theThickness << std::endl;

    //if (oldLayer != g.layer) {
    //  oldLayer = g.layer;
    //  std::cout << "new layer at " << i << (g.isBarrel ? " B  " :  (g.isPosZ ? " E+ " : " E- ")) << g.layer << " starting at " << g.rawId << std::endl;
    //}

    g.shiftX = 0.5f*p.lorentzShiftInCmX;
    g.shiftY = 0.5f*p.lorentzShiftInCmY;
    g.chargeWidthX = p.lorentzShiftInCmX * p.widthLAFractionX;
    g.chargeWidthY = p.lorentzShiftInCmY * p.widthLAFractionY;

    g.x0 = p.theOrigin.x();
    g.y0 = p.theOrigin.y();
    g.z0 = p.theOrigin.z();

    auto vv = p.theDet->surface().position();
    auto rr = pixelCPEforGPU::Rotation(p.theDet->surface().rotation());
    g.frame = pixelCPEforGPU::Frame(vv.x(),vv.y(),vv.z(),rr);


    // errors .....
   ClusterParamGeneric cp;
   auto gvx = p.theOrigin.x() + 40.f*m_commonParamsGPU.thePitchX;
   auto gvy = p.theOrigin.y();
   auto gvz = 1.f/p.theOrigin.z();
   //--- Note that the normalization is not required as only the ratio used

   // calculate angles
   cp.cotalpha = gvx*gvz;
   cp.cotbeta  = gvy*gvz;

   cp.with_track_angle = false;

   auto lape = p.theDet->localAlignmentError();
   if ( lape.invalid() ) lape = LocalError(); // zero....

#ifdef DUMP_ERRORS   
   auto m=10000.f;
   for (float qclus = 15000; qclus<35000; qclus+=15000){
     errorFromTemplates(p,cp,qclus);

     std::cout << i << ' ' << qclus << ' ' << cp.pixmx
               << ' ' << m*cp.sigmax << ' ' << m*cp.sx1 << ' ' << m*cp.sx2 
               << ' ' << m*cp.sigmay << ' ' << m*cp.sy1 << ' ' << m*cp.sy2
              << std::endl;
   }
   std::cout << i << ' ' << m*std::sqrt(lape.xx())  <<' '<< m*std::sqrt(lape.yy()) << std::endl;
#endif   

   
   errorFromTemplates(p,cp,20000.f);
   g.sx[0] = cp.sigmax;
   g.sx[1] = cp.sx1;
   g.sx[2] = cp.sx2;

   g.sy[0] = cp.sigmay;
   g.sy[1] = cp.sy1;
   g.sy[2] = cp.sy2;
   
   
   /*
    // from run1??
    if (i<96) {
      g.sx[0] = 0.00120;
      g.sx[1] = 0.00115;
      g.sx[2] = 0.0050;

      g.sy[0] = 0.00210;
      g.sy[1] = 0.00375;
      g.sy[2] = 0.0085;
    } else if (g.isBarrel) {
      g.sx[0] = 0.00120;
      g.sx[1] = 0.00115;
      g.sx[2] = 0.0050;

      g.sy[0] = 0.00210;
      g.sy[1] = 0.00375;
      g.sy[2] = 0.0085;
   } else {
      g.sx[0] = 0.0020;
      g.sx[1] = 0.0020;
      g.sx[2] = 0.0050;

      g.sy[0] = 0.0021;
      g.sy[1] = 0.0021;
      g.sy[2] = 0.0085;
   }
   */
   

   for (int i=0; i<3; ++i) {
     g.sx[i] = std::sqrt(g.sx[i]*g.sx[i]+lape.xx());
     g.sy[i] = std::sqrt(g.sy[i]*g.sy[i]+lape.yy());
   }

 }
}

PixelCPEFast::~PixelCPEFast() {}

PixelCPEFast::GPUData::~GPUData() {
  if(d_paramsOnGPU != nullptr) {
    cudaFree(h_paramsOnGPU.m_commonParams);
    cudaFree(h_paramsOnGPU.m_detParams);
    cudaFree(d_paramsOnGPU);
  }
}

PixelCPEBase::ClusterParam* PixelCPEFast::createClusterParam(const SiPixelCluster & cl) const
{
   return new ClusterParamGeneric(cl);
}



void
PixelCPEFast::errorFromTemplates(DetParam const & theDetParam, ClusterParamGeneric & theClusterParam, float qclus) const
{
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
   
   if ( UseErrorsFromTemplates_ ) {
      errorFromTemplates(theDetParam, theClusterParam, theClusterParam.theCluster->charge());
   } 
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
   
     // do GPU like ...
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
      auto xPos = cp.xpos[0];     
      auto yPos = cp.ypos[0];

   //--- Now put the two together
   LocalPoint pos_in_local( xPos, yPos );
   return pos_in_local;
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
