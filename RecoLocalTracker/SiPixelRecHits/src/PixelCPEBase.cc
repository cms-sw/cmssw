// Move geomCorrection to the concrete class. d.k. 06/06.
// Change drift direction. d.k. 06/06
// G. Giurgiu (ggiurgiu@pha.jhu.edu), 12/01/06, implemented the function: 
// computeAnglesFromDetPosition(const SiPixelCluster & cl, 
// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008
// Change to use Generic error & Template calibration from DB - D.Fehling 11/08


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/ProxyPixelTopology.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"

#define CORRECT_FOR_BIG_PIXELS

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"


#include <iostream>

using namespace std;

#define NEW_CPEERROR // must be constistent with base.cc, generic cc/h and genericProducer.cc 

namespace {
#ifndef NEW_CPEERROR  
  //const bool useNewSimplerErrors = true;
  const bool useNewSimplerErrors = false; // must be tha same as in generic 
#endif
}

//-----------------------------------------------------------------------------
//  A constructor run for generic and templates
//  
//-----------------------------------------------------------------------------
PixelCPEBase::PixelCPEBase(edm::ParameterSet const & conf, 
                           const MagneticField *mag, 
                           const TrackerGeometry& geom,
			   const TrackerTopology& ttopo,
			   const SiPixelLorentzAngle * lorentzAngle, 
			   const SiPixelGenErrorDBObject * genErrorDBObject, 
			   const SiPixelTemplateDBObject * templateDBobject,
			   const SiPixelLorentzAngle * lorentzAngleWidth,
			   int flag)
  //  : useLAAlignmentOffsets_(false), useLAOffsetFromConfig_(false),
  : useLAOffsetFromConfig_(false),
    useLAWidthFromConfig_(false), useLAWidthFromDB_(false), theFlag_(flag),
    magfield_(mag), geom_(geom), ttopo_(ttopo)
{

#ifdef EDM_ML_DEBUG
  nRecHitsTotal_=0;
  nRecHitsUsedEdge_=0,
#endif 
    
  //--- Lorentz angle tangent per Tesla
  lorentzAngle_ = lorentzAngle;
  lorentzAngleWidth_ = lorentzAngleWidth;
 
  //-- GenError Calibration Object (different from SiPixelCPEGenericErrorParm) from DB
  genErrorDBObject_ = genErrorDBObject;
  //cout<<" new errors "<<genErrorDBObject<<" "<<genErrorDBObject_<<endl;

  //-- Template Calibration Object from DB
#ifdef NEW_CPEERROR
  if(theFlag_!=0) templateDBobject_ = templateDBobject; // flag to check if it is generic or templates
#else
  templateDBobject_ = templateDBobject;
#endif

  // Configurables 
  // For both templates & generic 

  // Read templates and/or generic errors from DB
  LoadTemplatesFromDB_ = conf.getParameter<bool>("LoadTemplatesFromDB"); 
  //cout<<" use generros/templaets "<<LoadTemplatesFromDB_<<endl;

  //--- Algorithm's verbosity
  theVerboseLevel = 
    conf.getUntrackedParameter<int>("VerboseLevel",0);
  
  //-- Switch on/off E.B 
  alpha2Order = conf.getParameter<bool>("Alpha2Order");
  
  //--- A flag that could be used to change the behavior of
  //--- clusterProbability() in TSiPixelRecHit (the *transient* one).  
  //--- The problem is that the transient hits are made after the CPE runs
  //--- and they don't get the access to the PSet, so we pass it via the
  //--- CPE itself...
  //
  clusterProbComputationFlag_ 
    = (unsigned int) conf.getParameter<int>("ClusterProbComputationFlag");

  // This LA related parameters are only relevant for the Generic algo
  // They still have to be used in Base since the LA computation is in Base

  // Use LA-width from DB. 
  // If both (this and from config) are false LA-width is calcuated from LA-offset
  useLAWidthFromDB_ = conf.existsAs<bool>("useLAWidthFromDB")?
    conf.getParameter<bool>("useLAWidthFromDB"):false;

  // Use Alignment LA-offset in generic
  //useLAAlignmentOffsets_ = conf.existsAs<bool>("useLAAlignmentOffsets")?
  //conf.getParameter<bool>("useLAAlignmentOffsets"):false;

  // Used only for testing
  lAOffset_ = conf.existsAs<double>("lAOffset")?  // fixed LA value 
              conf.getParameter<double>("lAOffset"):0.0;
  lAWidthBPix_ = conf.existsAs<double>("lAWidthBPix")?   // fixed LA width 
                 conf.getParameter<double>("lAWidthBPix"):0.0;
  lAWidthFPix_ = conf.existsAs<double>("lAWidthFPix")?   // fixed LA width
                 conf.getParameter<double>("lAWidthFPix"):0.0;

  // Use LA-offset from config, for testing only
  if(lAOffset_>0.0) useLAOffsetFromConfig_ = true;
  // Use LA-width from config, split into fpix & bpix, for testing only
  if(lAWidthBPix_>0.0 || lAWidthFPix_>0.0) useLAWidthFromConfig_ = true;


  // For Templates only 
  // Compute the Lorentz shifts for this detector element for templates (from Alignment)
  DoLorentz_ = conf.existsAs<bool>("DoLorentz")?conf.getParameter<bool>("DoLorentz"):false;

  LogDebug("PixelCPEBase") <<" LA constants - "
			   <<lAOffset_<<" "<<lAWidthBPix_<<" "<<lAWidthFPix_<<endl; //dk
  
  fillDetParams();

  //cout<<" LA "<<lAOffset_<<" "<<lAWidthBPix_<<" "<<lAWidthFPix_<<endl; //dk
}

//-----------------------------------------------------------------------------
//  Fill all variables which are constant for an event (geometry)
//-----------------------------------------------------------------------------
void PixelCPEBase::fillDetParams()
{
  //cout<<" in fillDetParams "<<theFlag_<<endl;

  auto const & dus = geom_.detUnits();
  unsigned m_detectors = dus.size();
  for(unsigned int i=1;i<7;++i) {
    LogDebug("LookingForFirstStrip") << "Subdetector " << i 
				     << " GeomDetEnumerator " << GeomDetEnumerators::tkDetEnum[i] 
				     << " offset " << geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) 
				     << " is it strip? " << (geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() ? 
							     dus[geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isTrackerStrip() : false);
    if(geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() && 
       dus[geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isTrackerStrip()) {
      if(geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < m_detectors) m_detectors = geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
    }
  } 
  LogDebug("LookingForFirstStrip") << " Chosen offset: " << m_detectors;


  m_DetParams.resize(m_detectors);
  //cout<<"caching "<<m_detectors<<" pixel detectors"<<endl;
  for (unsigned i=0; i!=m_detectors;++i) {
    auto & p=m_DetParams[i];
    p.theDet = dynamic_cast<const PixelGeomDetUnit*>(dus[i]);
    assert(p.theDet); 
    assert(p.theDet->index()==int(i)); 

    p.theOrigin = p.theDet->surface().toLocal(GlobalPoint(0,0,0));
    
    //--- p.theDet->type() returns a GeomDetType, which implements subDetector()
    p.thePart = p.theDet->type().subDetector();
    
    //cout<<" in PixelCPEBase - in det "<<thePart<<endl; //dk

        //--- The location in of this DetUnit in a cyllindrical coord system (R,Z)
    //--- The call goes via BoundSurface, returned by p.theDet->surface(), but
    //--- position() is implemented in GloballyPositioned<> template
    //--- ( BoundSurface : Surface : GloballyPositioned<float> )
    //p.theDetR = p.theDet->surface().position().perp();  //Not used, AH
    //p.theDetZ = p.theDet->surface().position().z();  //Not used, AH
    //--- Define parameters for chargewidth calculation
    
    //--- bounds() is implemented in BoundSurface itself.
    p.theThickness = p.theDet->surface().bounds().thickness();
    
    // Cache the det id for templates and generic erros 

    if(theFlag_==0) { // for generic
#ifdef NEW_CPEERROR
      if(LoadTemplatesFromDB_ ) // do only if genError requested 
	p.detTemplateId = genErrorDBObject_->getGenErrorID(p.theDet->geographicalId().rawId());
#else   
      if(useNewSimplerErrors) 
	p.detTemplateId = genErrorDBObject_->getGenErrorID(p.theDet->geographicalId().rawId());
      else 
        p.detTemplateId = templateDBobject_->getTemplateID(p.theDet->geographicalId().rawId());
#endif
    } else {          // for templates
      p.detTemplateId = templateDBobject_->getTemplateID(p.theDet->geographicalId());
    }

    // just for testing
    //int i1 = 0;
    //if(theFlag_==0) i1 = genErrorDBObject_->getGenErrorID(p.theDet->geographicalId().rawId());
    //int i2= templateDBobject_->getTemplateID(p.theDet->geographicalId().rawId());
    //int i3= templateDBobject_->getTemplateID(p.theDet->geographicalId());
    //if(i2!=i3) cout<<i2<<" != "<<i3<<endl;
    //cout<<i<<" "<<p.detTemplateId<<" "<<i1<<" "<<i2<<" "<<i3<<endl;
    
    auto topol = &(p.theDet->specificTopology());
       p.theTopol=topol;
       auto const proxyT = dynamic_cast<const ProxyPixelTopology*>(p.theTopol);
       if (proxyT) p.theRecTopol = dynamic_cast<const RectangularPixelTopology*>(&(proxyT->specificTopology()));
       else p.theRecTopol = dynamic_cast<const RectangularPixelTopology*>(p.theTopol);
       assert(p.theRecTopol);
       
       //---- The geometrical description of one module/plaquette
       //p.theNumOfRow = p.theRecTopol->nrows();	// rows in x //Not used, AH
       //p.theNumOfCol = p.theRecTopol->ncolumns();	// cols in y //Not used, AH
       std::pair<float,float> pitchxy = p.theRecTopol->pitch();
       p.thePitchX = pitchxy.first;	     // pitch along x
       p.thePitchY = pitchxy.second;	     // pitch along y
     
    //p.theSign = isFlipped(&p) ? -1 : 1; //Not used, AH

    LocalVector Bfield = p.theDet->surface().toLocal(magfield_->inTesla(p.theDet->surface().position()));
    p.bz = Bfield.z();


    // Compute the Lorentz shifts for this detector element
    if ( (theFlag_==0) || DoLorentz_ ) {  // do always for generic and if(DOLorentz) for templates
      p.driftDirection = driftDirection(p, Bfield );
      computeLorentzShifts(p);
    }


    LogDebug("PixelCPEBase") << "***** PIXEL LAYOUT *****" 
			     << " thePart = " << p.thePart
			     << " theThickness = " << p.theThickness
			     << " thePitchX  = " << p.thePitchX 
			     << " thePitchY  = " << p.thePitchY; 
    //			     << " theLShiftX  = " << p.theLShiftX;
    
    
      }
}

//-----------------------------------------------------------------------------
//  One function to cache the variables common for one DetUnit.
//-----------------------------------------------------------------------------
void
PixelCPEBase::setTheClu( DetParam const & theDetParam, ClusterParam & theClusterParam ) const 
{

  //--- Geometric Quality Information
  int minInX,minInY,maxInX,maxInY=0;
  minInX = theClusterParam.theCluster->minPixelRow();
  minInY = theClusterParam.theCluster->minPixelCol();
  maxInX = theClusterParam.theCluster->maxPixelRow();
  maxInY = theClusterParam.theCluster->maxPixelCol();
  
  theClusterParam.isOnEdge_ = theDetParam.theRecTopol->isItEdgePixelInX(minInX) | theDetParam.theRecTopol->isItEdgePixelInX(maxInX) |
    theDetParam.theRecTopol->isItEdgePixelInY(minInY) | theDetParam.theRecTopol->isItEdgePixelInY(maxInY) ;
  
  // FOR NOW UNUSED. KEEP IT IN CASE WE WANT TO USE IT IN THE FUTURE  
  // Bad Pixels have their charge set to 0 in the clusterizer 
  //hasBadPixels_ = false;
  //for(unsigned int i=0; i<theClusterParam.theCluster->pixelADC().size(); ++i) {
  //if(theClusterParam.theCluster->pixelADC()[i] == 0) { hasBadPixels_ = true; break;}
  //}
  
  theClusterParam.spansTwoROCs_ = theDetParam.theRecTopol->containsBigPixelInX(minInX,maxInX) |
    theDetParam.theRecTopol->containsBigPixelInY(minInY,maxInY);

}


//-----------------------------------------------------------------------------
//  Compute alpha_ and beta_ from the LocalTrajectoryParameters.
//  Note: should become const after both localParameters() become const.
//-----------------------------------------------------------------------------
void PixelCPEBase::
computeAnglesFromTrajectory( DetParam const & theDetParam, ClusterParam & theClusterParam,
			     const LocalTrajectoryParameters & ltp) const
{
  //cout<<" in PixelCPEBase:computeAnglesFromTrajectory - "<<endl; //dk

  //theClusterParam.loc_traj_param = ltp;

  LocalVector localDir = ltp.momentum();
  
  
  float locx = localDir.x();
  float locy = localDir.y();
  float locz = localDir.z();
  
  /*
  // Danek's definition 
  alpha_ = acos(locx/sqrt(locx*locx+locz*locz));
  if ( isFlipped() )                    // &&& check for FPIX !!!
  alpha_ = PI - alpha_ ;
  beta_ = acos(locy/sqrt(locy*locy+locz*locz));
  */
  
  
  theClusterParam.cotalpha = locx/locz;
  theClusterParam.cotbeta  = locy/locz;
  //theClusterParam.zneg = (locz < 0); // Not used, AH
  
  
  LocalPoint trk_lp = ltp.position();
  theClusterParam.trk_lp_x = trk_lp.x();
  theClusterParam.trk_lp_y = trk_lp.y();
  
  theClusterParam.with_track_angle = true;


  // ggiurgiu@jhu.edu 12/09/2010 : needed to correct for bows/kinks
  AlgebraicVector5 vec_trk_parameters = ltp.mixedFormatVector();
  theClusterParam.loc_trk_pred = Topology::LocalTrackPred( vec_trk_parameters );
  
}

//-----------------------------------------------------------------------------
//  Estimate theAlpha for barrel, based on the det position.
//  &&& Needs to be consolidated from the above.
//-----------------------------------------------------------------------------
//float 
//PixelCPEBase::estimatedAlphaForBarrel(float centerx) const
//{
//  float tanalpha = theSign * (centerx-theOffsetX) * thePitchX / theDetR;
//  return PI/2.0 - atan(tanalpha);
//}




//-----------------------------------------------------------------------------
//  Compute alpha_ and beta_ from the position of this DetUnit.
//  &&& DOES NOT WORK FOR NOW. d.k. 6/06
// The angles from dets are calculated internaly in the PixelCPEInitial class
//-----------------------------------------------------------------------------
// G. Giurgiu, 12/01/06 : implement the function
void PixelCPEBase::
computeAnglesFromDetPosition(DetParam const & theDetParam, ClusterParam & theClusterParam ) const
{
 
  
  /*
  // get cluster center of gravity (of charge)
  float xcenter = cl.x();
  float ycenter = cl.y();
  
  // get the cluster position in local coordinates (cm)

  // ggiurgiu@jhu.edu 12/09/2010 : This function is called without track info, therefore there are no track 
  // angles to provide here. Call the default localPosition (without track info)
  LocalPoint lp = theTopol->localPosition( MeasurementPoint(xcenter, ycenter) );


  // get the cluster position in global coordinates (cm)
  GlobalPoint gp = theDet->surface().toGlobal( lp );
  float gp_mod = sqrt( gp.x()*gp.x() + gp.y()*gp.y() + gp.z()*gp.z() );

  // normalize
  float gpx = gp.x()/gp_mod;
  float gpy = gp.y()/gp_mod;
  float gpz = gp.z()/gp_mod;

  // make a global vector out of the global point; this vector will point from the 
  // origin of the detector to the cluster
  GlobalVector gv(gpx, gpy, gpz);

  // make local unit vector along local X axis
  const Local3DVector lvx(1.0, 0.0, 0.0);

  // get the unit X vector in global coordinates/
  GlobalVector gvx = theDet->surface().toGlobal( lvx );

  // make local unit vector along local Y axis
  const Local3DVector lvy(0.0, 1.0, 0.0);

  // get the unit Y vector in global coordinates
  GlobalVector gvy = theDet->surface().toGlobal( lvy );
   
  // make local unit vector along local Z axis
  const Local3DVector lvz(0.0, 0.0, 1.0);

  // get the unit Z vector in global coordinates
  GlobalVector gvz = theDet->surface().toGlobal( lvz );
    
  // calculate the components of gv (the unit vector pointing to the cluster) 
  // in the local coordinate system given by the basis {gvx, gvy, gvz}
  // note that both gv and the basis {gvx, gvy, gvz} are given in global coordinates
  float gv_dot_gvx = gv.x()*gvx.x() + gv.y()*gvx.y() + gv.z()*gvx.z();
  float gv_dot_gvy = gv.x()*gvy.x() + gv.y()*gvy.y() + gv.z()*gvy.z();
  float gv_dot_gvz = gv.x()*gvz.x() + gv.y()*gvz.y() + gv.z()*gvz.z();
  */
  
  // all the above is equivalent to 
  LocalPoint lp = theDetParam.theTopol->localPosition( MeasurementPoint(theClusterParam.theCluster->x(), theClusterParam.theCluster->y()) );
  auto gvx = lp.x()-theDetParam.theOrigin.x();
  auto gvy = lp.y()-theDetParam.theOrigin.y();
  auto gvz = -1.f/theDetParam.theOrigin.z();
  //  normalization not required as only ratio used... 
  

  //theClusterParam.zneg = (gvz < 0); // Not used, AH

  // calculate angles
  theClusterParam.cotalpha = gvx*gvz;
  theClusterParam.cotbeta  = gvy*gvz;

  theClusterParam.with_track_angle = false;


  /*
  // used only in dberror param...
  auto alpha = HALF_PI - std::atan(cotalpha_);
  auto beta = HALF_PI - std::atan(cotbeta_); 
  if (zneg) { beta -=PI; alpha -=PI;}

  auto alpha_ = atan2( gv_dot_gvz, gv_dot_gvx );
  auto beta_  = atan2( gv_dot_gvz, gv_dot_gvy );

  assert(std::abs(std::round(alpha*10000.f)-std::round(alpha_*10000.f))<2);
  assert(std::abs(std::round(beta*10000.f)-std::round(beta_*10000.f))<2);
  */

}



//-----------------------------------------------------------------------------
// The isFlipped() is a silly way to determine which detectors are inverted.
// In the barrel for every 2nd ladder the E field direction is in the
// global r direction (points outside from the z axis), every other
// ladder has the E field inside. Something similar is in the 
// forward disks (2 sides of the blade). This has to be recognised
// because the charge sharing effect is different.
//
// The isFliped does it by looking and the relation of the local (z always
// in the E direction) to global coordinates. There is probably a much 
// better way.
//-----------------------------------------------------------------------------
bool PixelCPEBase::isFlipped(DetParam const & theDetParam) const 
{
  // Check the relative position of the local +/- z in global coordinates.
  float tmp1 = theDetParam.theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp2();
  float tmp2 = theDetParam.theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp2();
  //cout << " 1: " << tmp1 << " 2: " << tmp2 << endl;
  if ( tmp2<tmp1 ) return true;
  else return false;    
}
//------------------------------------------------------------------------
PixelCPEBase::DetParam const & PixelCPEBase::detParam(const GeomDetUnit & det) const {
  auto i = det.index();
  //cout << "get parameters of detector " << i << endl;
  assert(i<int(m_DetParams.size()));
  //if (i>=int(m_DetParams.size())) m_DetParams.resize(i+1);  // should never happen!
  const DetParam & p = m_DetParams[i];
  return p;
}

//-----------------------------------------------------------------------------
//  Drift direction.
//  Works OK for barrel and forward.
//  The formulas used for dir_x,y,z have to be exactly the same as the ones
//  used in the digitizer (SiPixelDigitizerAlgorithm.cc).
//
//-----------------------------------------------------------------------------
LocalVector 
PixelCPEBase::driftDirection(DetParam & theDetParam, GlobalVector bfield ) const {

  Frame detFrame(theDetParam.theDet->surface().position(), theDetParam.theDet->surface().rotation());
  LocalVector Bfield = detFrame.toLocal(bfield);
  return driftDirection(theDetParam,Bfield);
  
}

LocalVector 
PixelCPEBase::driftDirection(DetParam & theDetParam, LocalVector Bfield ) const {
  const bool LocalPrint = false;

  // Use LA from DB or from config 
  float langle = 0.;
  if( !useLAOffsetFromConfig_ ) {  // get it from DB
    if(lorentzAngle_ != NULL) {  // a real LA object 
      langle = lorentzAngle_->getLorentzAngle(theDetParam.theDet->geographicalId().rawId());
      //cout<<" la "<<langle<<" "<< theDetParam.theDet->geographicalId().rawId() <<endl;
    } else { // no LA, unused 
      //cout<<" LA object is NULL, assume LA = 0"<<endl; //dk
      langle = 0; // set to a fake value
    }
    if(LocalPrint) cout<<" Will use LA Offset from DB "<<langle<<endl;
  } else {  // from config file 
    langle = lAOffset_;
    if(LocalPrint) cout<<" Will use LA Offset from config "<<langle<<endl;
  } 
    
  // Now do the LA width stuff 
  theDetParam.widthLAFractionX = 1.; // predefine to 1 (default) if things fail
  theDetParam.widthLAFractionY = 1.;

  // Compute the charge width, generic only
  if(theFlag_==0) {
      
    if(useLAWidthFromDB_ && (lorentzAngleWidth_ != NULL) ) {  
      // take it from a seperate, special LA DB object (forWidth)
      
      auto langleWidth = lorentzAngleWidth_->getLorentzAngle(theDetParam.theDet->geographicalId().rawId());	  
      if(langleWidth!=0.0) theDetParam.widthLAFractionX = std::abs(langleWidth/langle);
      // leave the widthLAFractionY=1.
      //cout<<" LAWidth lorentz width "<<theDetParam.widthLAFractionX<<" "<<theDetParam.widthLAFractionY<<endl;
            
    } else if(useLAWidthFromConfig_) { // get from config 
      
      double lAWidth=0;
      if( GeomDetEnumerators::isTrackerPixel(theDetParam.thePart) && GeomDetEnumerators::isBarrel(theDetParam.thePart) ) lAWidth = lAWidthBPix_; // barrel
      else lAWidth = lAWidthFPix_;
      
      if(langle!=0.0) theDetParam.widthLAFractionX = std::abs(lAWidth/langle);
      // fix the FractionY at 1
      
      //cout<<" Lorentz width from config "<<theDetParam.widthLAFractionX<<" "<<theDetParam.widthLAFractionY<<endl;
      
    } else { // get if from the offset LA (old method used until 2013)
      // do nothing      
      //cout<<" Old default LA width method "<<theDetParam.widthLAFractionX<<" "<<theDetParam.widthLAFractionY<<endl;
      
    }
    
    //cout<<" Final LA fraction  "<<theDetParam.widthLAFractionX<<" "<<theDetParam.widthLAFractionY<<endl;
    
  }  // if flag 


  if(LocalPrint) cout<<" in PixelCPEBase:driftDirection - "<<langle<<" "<<Bfield<<endl; //dk

  float alpha2 = alpha2Order ?  langle*langle : 0; // 

  // **********************************************************************
  // Our convention is the following:
  // +x is defined by the direction of the Lorentz drift!
  // +z is defined by the direction of E field (so electrons always go into -z!)
  // +y is defined by +x and +z, and it turns out to be always opposite to the +B field.
  // **********************************************************************
  
  float dir_x = -( langle * Bfield.y() + alpha2* Bfield.z()* Bfield.x() );
  float dir_y =  ( langle * Bfield.x() - alpha2* Bfield.z()* Bfield.y() );
  float dir_z = -( 1.f + alpha2* Bfield.z()*Bfield.z() );
  auto scale = 1.f/std::abs( dir_z );  // same as 1 + alpha2*Bfield.z()*Bfield.z()
  LocalVector  dd(dir_x*scale, dir_y*scale, -1.f );  // last is -1 !

  LogDebug("PixelCPEBase") << " The drift direction in local coordinate is "  << dd   ;
	
  return dd;
}

//-----------------------------------------------------------------------------
//  One-shot computation of the driftDirection and both lorentz shifts
//-----------------------------------------------------------------------------
void
PixelCPEBase::computeLorentzShifts(DetParam & theDetParam) const {

  //cout<<" in PixelCPEBase:computeLorentzShifts - "<<driftDirection_<<endl; //dk

  // Max shift (at the other side of the sensor) in cm 
  theDetParam.lorentzShiftInCmX = theDetParam.driftDirection.x()/theDetParam.driftDirection.z() * theDetParam.theThickness;  // 
  theDetParam.lorentzShiftInCmY = theDetParam.driftDirection.y()/theDetParam.driftDirection.z() * theDetParam.theThickness;  //
  
  //cout<<" in PixelCPEBase:computeLorentzShifts - "
  //<<lorentzShiftInCmX_<<" "
  //<<lorentzShiftInCmY_<<" "
  //<<endl; //dk
  
  LogDebug("PixelCPEBase") << " The drift direction in local coordinate is " 
			   << theDetParam.driftDirection    ;
}

//-----------------------------------------------------------------------------
//! A convenience method to fill a whole SiPixelRecHitQuality word in one shot.
//! This way, we can keep the details of what is filled within the pixel
//! code and not expose the Transient SiPixelRecHit to it as well.  The name
//! of this function is chosen to match the one in SiPixelRecHit.
//-----------------------------------------------------------------------------
SiPixelRecHitQuality::QualWordType 
PixelCPEBase::rawQualityWord(ClusterParam & theClusterParam) const
{
  SiPixelRecHitQuality::QualWordType qualWord(0);
  float probabilityXY;
  if ( theClusterParam.probabilityX_ !=0 && theClusterParam.probabilityY_ !=0 ) 
     probabilityXY = theClusterParam.probabilityX_ * theClusterParam.probabilityY_ * (1.f - std::log(theClusterParam.probabilityX_ * theClusterParam.probabilityY_) ) ;
  else 
     probabilityXY = 0;
  SiPixelRecHitQuality::thePacking.setProbabilityXY ( probabilityXY ,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setProbabilityQ  ( theClusterParam.probabilityQ_ , 
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setQBin          ( (int)theClusterParam.qBin_, 
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setIsOnEdge      ( theClusterParam.isOnEdge_,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setHasBadPixels  ( theClusterParam.hasBadPixels_,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setSpansTwoROCs  ( theClusterParam.spansTwoROCs_,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setHasFilledProb ( theClusterParam.hasFilledProb_,
                                                      qualWord );
  
  return qualWord;
}
