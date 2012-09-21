// Move geomCorrection to the concrete class. d.k. 06/06.
// Change drift direction. d.k. 06/06

// G. Giurgiu (ggiurgiu@pha.jhu.edu), 12/01/06, implemented the function: 
// computeAnglesFromDetPosition(const SiPixelCluster & cl, 
//			        const GeomDetUnit    & det ) const
//                                    09/09/07, replaced assert statements with throw cms::Exception 
//                                              and fix an invalid pointer check in setTheDet function 
//                                    09/21/07, implement caching of Lorentz drift direction
//                                    01/24/09, use atan2 to get the alpha and beta angles
// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008
// Change to use Generic error & Template calibration from DB - D.Fehling 11/08


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/ProxyPixelTopology.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEBase.h"

//#define TPDEBUG
#define CORRECT_FOR_BIG_PIXELS

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include <iostream>

using namespace std;

const float PI = 3.141593;
const float degsPerRad = 57.29578;

//-----------------------------------------------------------------------------
//  A fairly boring constructor.  All quantities are DetUnit-dependent, and
//  will be initialized in setTheDet().
//-----------------------------------------------------------------------------
PixelCPEBase::PixelCPEBase(edm::ParameterSet const & conf, const MagneticField *mag, const SiPixelLorentzAngle * lorentzAngle, 
			   const SiPixelCPEGenericErrorParm * genErrorParm, const SiPixelTemplateDBObject * templateDBobject)
  : theDet(nullptr), theTopol(nullptr), theRecTopol(nullptr), theParam(nullptr), nRecHitsTotal_(0), nRecHitsUsedEdge_(0),
    probabilityX_(0.0), probabilityY_(0.0),
    probabilityQ_(0.0), qBin_(0),
    isOnEdge_(false), hasBadPixels_(false),
    spansTwoROCs_(false), hasFilledProb_(false),
    loc_trk_pred_(0.0, 0.0, 0.0, 0.0)
{
  //--- Lorentz angle tangent per Tesla

  lorentzAngle_ = lorentzAngle;
 
  //--- Algorithm's verbosity
  theVerboseLevel = 
    conf.getUntrackedParameter<int>("VerboseLevel",0);
  
  //-- Magnetic Field
  magfield_ = mag;
  
  //-- Error Parametriaztion from DB for CPE Generic
  genErrorParm_ = genErrorParm;
  
  //-- Template Calibration Object from DB
  templateDBobject_ = templateDBobject;
  
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
  
}

//-----------------------------------------------------------------------------
//  One function to cache the variables common for one DetUnit.
//-----------------------------------------------------------------------------
void
PixelCPEBase::setTheDet( const GeomDetUnit & det, const SiPixelCluster & cluster ) const 
{
  if ( theDet == &det )
    return;       // we have already seen this det unit
  
  //--- This is a new det unit, so cache it
  theDet = dynamic_cast<const PixelGeomDetUnit*>( &det );

  if ( !theDet ) 
    {
      throw cms::Exception(" PixelCPEBase::setTheDet : ")
            << " Wrong pointer to PixelGeomDetUnit object !!!";
    }

  //--- theDet->type() returns a GeomDetType, which implements subDetector()
  thePart = theDet->type().subDetector();
  switch ( thePart ) 
    {
    case GeomDetEnumerators::PixelBarrel:
      // A barrel!  A barrel!
      break;
    case GeomDetEnumerators::PixelEndcap:
      // A forward!  A forward!
      break;
    default:
      throw cms::Exception("PixelCPEBase::setTheDet :")
      	<< "PixelCPEBase: A non-pixel detector type in here?" ;
    }

  //--- The location in of this DetUnit in a cyllindrical coord system (R,Z)
  //--- The call goes via BoundSurface, returned by theDet->surface(), but
  //--- position() is implemented in GloballyPositioned<> template
  //--- ( BoundSurface : Surface : GloballyPositioned<float> )
  theDetR = theDet->surface().position().perp();
  theDetZ = theDet->surface().position().z();
  //--- Define parameters for chargewidth calculation

  //--- bounds() is implemented in BoundSurface itself.
  theThickness = theDet->surface().bounds().thickness();

  //--- Cache the topology.
  // ggiurgiu@jhu.edu 12/09/2010 : no longer need to dynamyc cast to RectangularPixelTopology
  //theTopol
  //= dynamic_cast<const RectangularPixelTopology*>( & (theDet->specificTopology()) );

  auto topol = &(theDet->specificTopology());
  if unlikely(topol!=theTopol) { // there is ONE topology!)
      theTopol=topol;
      auto const proxyT = dynamic_cast<const ProxyPixelTopology*>(theTopol);
      if (proxyT) theRecTopol = dynamic_cast<const RectangularPixelTopology*>(&(proxyT->specificTopology()));
      else theRecTopol = dynamic_cast<const RectangularPixelTopology*>(theTopol);
      assert(theRecTopol);
      
      //---- The geometrical description of one module/plaquette
      theNumOfRow = theRecTopol->nrows();      // rows in x
      theNumOfCol = theRecTopol->ncolumns();   // cols in y
      std::pair<float,float> pitchxy = theRecTopol->pitch();
      thePitchX = pitchxy.first;            // pitch along x
      thePitchY = pitchxy.second;           // pitch along y
    }
  
  theSign = isFlipped() ? -1 : 1;


  // will cache if not yest there (need some of the above)
  theParam = &param();

  // this "has wrong sign..."
  driftDirection_ = (*theParam).drift;
 

  //--- The Lorentz shift.
  theLShiftX = lorentzShiftX();

  theLShiftY = lorentzShiftY();

  // testing 
  if(thePart == GeomDetEnumerators::PixelBarrel) {
    //cout<<" lorentz shift "<<theLShiftX<<" "<<theLShiftY<<endl;
    theLShiftY=0.;
  }

  //--- Geometric Quality Information
  int minInX,minInY,maxInX,maxInY=0;
  minInX = cluster.minPixelRow();
  minInY = cluster.minPixelCol();
  maxInX = cluster.maxPixelRow();
  maxInY = cluster.maxPixelCol();
  
  if(theRecTopol->isItEdgePixelInX(minInX) || theRecTopol->isItEdgePixelInX(maxInX) ||
     theRecTopol->isItEdgePixelInY(minInY) || theRecTopol->isItEdgePixelInY(maxInY) )  {
    isOnEdge_ = true;
  }
  else isOnEdge_ = false;
  
  // Bad Pixels have their charge set to 0 in the clusterizer 
  hasBadPixels_ = false;
  for(unsigned int i=0; i<cluster.pixelADC().size(); ++i) {
    if(cluster.pixelADC()[i] == 0) hasBadPixels_ = true;
  }
  
  if(theRecTopol->containsBigPixelInX(minInX,maxInX) ||
     theRecTopol->containsBigPixelInY(minInY,maxInY) )  {
    spansTwoROCs_ = true;
  }
  else spansTwoROCs_ = false;
  
  
  if (theVerboseLevel > 1) 
    {
      LogDebug("PixelCPEBase") << "***** PIXEL LAYOUT *****" 
			       << " thePart = " << thePart
			       << " theThickness = " << theThickness
			       << " thePitchX  = " << thePitchX 
			       << " thePitchY  = " << thePitchY 
	// << " theOffsetX = " << theOffsetX 
	// << " theOffsetY = " << theOffsetY 
			       << " theLShiftX  = " << theLShiftX;
    }
  
}


//-----------------------------------------------------------------------------
//  Compute alpha_ and beta_ from the LocalTrajectoryParameters.
//  Note: should become const after both localParameters() become const.
//-----------------------------------------------------------------------------
void PixelCPEBase::
computeAnglesFromTrajectory( const SiPixelCluster & cl,
			     const GeomDetUnit    & det, 
			     const LocalTrajectoryParameters & ltp) const
{
  loc_traj_param_ = ltp;

  LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
  
  // &&& Or, maybe we need to move to the local frame ???
  //  LocalVector localDir( theDet->toLocal(theState.globalDirection()));
  //thePart = theDet->type().part();
  
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

  // &&& In the above, why not use atan2() ?
  // ggiurgiu@fnal.gov, 01/24/09 :  Use it now.
  alpha_ = atan2( locz, locx );
  beta_  = atan2( locz, locy );

  cotalpha_ = locx/locz;
  cotbeta_  = locy/locz;

  LocalPoint trk_lp = ltp.position();
  trk_lp_x = trk_lp.x();
  trk_lp_y = trk_lp.y();
  
  with_track_angle = true;


  // ggiurgiu@jhu.edu 12/09/2010 : needed to correct for bows/kinks
  AlgebraicVector5 vec_trk_parameters = ltp.mixedFormatVector();
  //loc_trk_pred = &Topology::LocalTrackPred( vec_trk_parameters );
  loc_trk_pred_ = Topology::LocalTrackPred( vec_trk_parameters );
  
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
//  The local position.
// Should do correctly the big pixels.
// who is using this version????
//-----------------------------------------------------------------------------
LocalPoint
PixelCPEBase::localPosition( const SiPixelCluster& cluster, 
			     const GeomDetUnit & det) const {
  setTheDet( det, cluster );
  
  float lpx = xpos(cluster);
  float lpy = ypos(cluster);
  float lxshift = theLShiftX * thePitchX;  // shift in cm
  float lyshift = theLShiftY * thePitchY;
  LocalPoint cdfsfs(lpx-lxshift, lpy-lyshift);
  return cdfsfs;
}

//-----------------------------------------------------------------------------
//  Seems never used?
//-----------------------------------------------------------------------------
MeasurementPoint 
PixelCPEBase::measurementPosition( const SiPixelCluster& cluster, 
				   const GeomDetUnit & det) const {

  LocalPoint lp = localPosition(cluster,det);

  // ggiurgiu@jhu.edu 12/09/2010 : trk angles needed for bow/kink correction

  if ( with_track_angle )
    return theTopol->measurementPosition( lp, Topology::LocalTrackAngles( loc_traj_param_.dxdz(), loc_traj_param_.dydz() ) );
  else 
    return theTopol->measurementPosition( lp );

}


//-----------------------------------------------------------------------------
//  Once we have the position, feed it to the topology to give us
//  the error.  
//  &&& APPARENTLY THIS METHOD IS NOT BEING USED ??? (Delete it?)
//-----------------------------------------------------------------------------
MeasurementError  
PixelCPEBase::measurementError( const SiPixelCluster& cluster, const GeomDetUnit & det) const 
{
  LocalPoint lp( localPosition(cluster, det) );
  LocalError le( localError(   cluster, det) );

  // ggiurgiu@jhu.edu 12/09/2010 : trk angles needed for bow/kink correction
  if ( with_track_angle )
    return theTopol->measurementError( lp, le, Topology::LocalTrackAngles( loc_traj_param_.dxdz(), loc_traj_param_.dydz() ) );
  else 
    return theTopol->measurementError( lp, le );
}

//-----------------------------------------------------------------------------
//  Compute alpha_ and beta_ from the position of this DetUnit.
//  &&& DOES NOT WORK FOR NOW. d.k. 6/06
// The angles from dets are calculated internaly in the PixelCPEInitial class
//-----------------------------------------------------------------------------
// G. Giurgiu, 12/01/06 : implement the function
void PixelCPEBase::
computeAnglesFromDetPosition(const SiPixelCluster & cl, 
			     const GeomDetUnit    & det ) const
{
  //--- This is a new det unit, so cache it
  theDet = dynamic_cast<const PixelGeomDetUnit*>( &det );
  if ( ! theDet ) 
    {
      throw cms::Exception("PixelCPEBase::computeAngleFromDetPosition")
	<< " Wrong pointer to pixel detector !!!" << endl;
    
    }

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


  /* all the above is equivalent to  
     const Local3DPoint origin =   theDet->surface().toLocal(GlobalPoint(0,0,0)); // can be computed once...
     auto gvx = lp.x()-origin.x();
     auto gvy = lp.y()-origin.y();
     auto gvz = -origin.z();
  *  normalization not required as only ratio used... 
  */


  // calculate angles
  alpha_ = atan2( gv_dot_gvz, gv_dot_gvx );
  beta_  = atan2( gv_dot_gvz, gv_dot_gvy );

  cotalpha_ = gv_dot_gvx / gv_dot_gvz;
  cotbeta_  = gv_dot_gvy / gv_dot_gvz;

  with_track_angle = false;
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
bool PixelCPEBase::isFlipped() const 
{
  // Check the relative position of the local +/- z in global coordinates.
  float tmp1 = theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp2();
  float tmp2 = theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp2();
  //cout << " 1: " << tmp1 << " 2: " << tmp2 << endl;
  if ( tmp2<tmp1 ) return true;
  else return false;    
}

PixelCPEBase::Param const & PixelCPEBase::param() const {
  Param & p = m_Params[ theDet->geographicalId().rawId() ];
  if unlikely ( p.bz<-1.e10f  ) { 
      LocalVector Bfield = theDet->surface().toLocal(magfield_->inTesla(theDet->surface().position()));
      p.drift = driftDirection(Bfield );
      p.bz = Bfield.z();
    }
  return p;
}


//-----------------------------------------------------------------------------
// HALF OF the Lorentz shift (so for the full shift multiply by 2), and
// in the units of pitch.  (So note these are neither local nor measurement
// units!)
//-----------------------------------------------------------------------------
float PixelCPEBase::lorentzShiftX() const 
{
  LocalVector dir = getDrift();

  // max shift in cm 
  float xdrift = dir.x()/dir.z() * theThickness;  
  // express the shift in units of pitch, 
  // divide by 2 to get the average correction
  float lshift = xdrift / (thePitchX*2.); 
    
  return lshift;  
  

}

float PixelCPEBase::lorentzShiftY() const 
{
 
  LocalVector dir = getDrift();
  
  float ydrift = dir.y()/dir.z() * theThickness;
  float lshift = ydrift / (thePitchY * 2.f);
  return lshift; 
  

}


//-----------------------------------------------------------------------------
//  Drift direction.
//  Works OK for barrel and forward.
//  The formulas used for dir_x,y,z have to be exactly the same as the ones
//  used in the digitizer (SiPixelDigitizerAlgorithm.cc).
//  Assumption: setTheDet() has been called already.
//
//  Petar (2/23/07): uhm, actually, there is a bug in the sign for both X and Y!
//  (The signs have been fixed in SiPixelDigitizer, but not in here.)
//-----------------------------------------------------------------------------
LocalVector 
PixelCPEBase::driftDirection( GlobalVector bfield ) const {

  Frame detFrame(theDet->surface().position(), theDet->surface().rotation());
  LocalVector Bfield = detFrame.toLocal(bfield);
  return driftDirection(Bfield);
  
}

LocalVector 
PixelCPEBase::driftDirection( LocalVector Bfield ) const {
  
  if(lorentzAngle_ == 0){
    throw cms::Exception("invalidPointer") << "[PixelCPEBase::driftDirection] zero pointer to lorentz angle record ";
  }
  double langle = lorentzAngle_->getLorentzAngle(theDet->geographicalId().rawId());
  float alpha2;
  if (alpha2Order) {
    alpha2 = langle*langle;
  } else {
    alpha2 = 0.0;
  }
  // &&& dir_x should have a "-" and dir_y a "+"
  // **********************************************************************
  // Our convention is the following:
  // +x is defined by the direction of the Lorentz drift!
  // +z is defined by the direction of E field (so electrons always go into -z!)
  // +y is defined by +x and +z, and it turns out to be always opposite to the +B field.
  // **********************************************************************
  
  float dir_x =  ( langle * Bfield.y() + alpha2* Bfield.z()* Bfield.x() );
  float dir_y = -( langle * Bfield.x() - alpha2* Bfield.z()* Bfield.y() );
  float dir_z = -( 1.f + alpha2* Bfield.z()*Bfield.z() );
  double scale = 1.f/std::abs( dir_z );  // same as 1 + alpha2*Bfield.z()*Bfield.z()
  LocalVector  dd(dir_x*scale, dir_y*scale, -1.f );  // last is -1 !
  if ( theVerboseLevel > 9 ) 
    LogDebug("PixelCPEBase") << " The drift direction in local coordinate is " 
			     << dd   ;
	
  return dd;
}

//-----------------------------------------------------------------------------
//  One-shot computation of the driftDirection and both lorentz shifts
//-----------------------------------------------------------------------------
void
PixelCPEBase::computeLorentzShifts() const 
{
  // this "has wrong sign..."  so "corrected below
   driftDirection_ = getDrift();
 
 // Max shift (at the other side of the sensor) in cm 
  lorentzShiftInCmX_ = -driftDirection_.x()/driftDirection_.z() * theThickness;  // &&& redundant
  // Express the shift in units of pitch, 
  lorentzShiftX_ = lorentzShiftInCmX_ / thePitchX ; 
   
  // Max shift (at the other side of the sensor) in cm 
  lorentzShiftInCmY_ = -driftDirection_.y()/driftDirection_.z() * theThickness;  // &&& redundant
  // Express the shift in units of pitch, 
  lorentzShiftY_ = lorentzShiftInCmY_ / thePitchY;


  if ( theVerboseLevel > 9 ) {
    LogDebug("PixelCPEBase") << " The drift direction in local coordinate is " 
			     << driftDirection_    ;
    
  }
}

//-----------------------------------------------------------------------------
//! A convenience method to fill a whole SiPixelRecHitQuality word in one shot.
//! This way, we can keep the details of what is filled within the pixel
//! code and not expose the Transient SiPixelRecHit to it as well.  The name
//! of this function is chosen to match the one in SiPixelRecHit.
//-----------------------------------------------------------------------------
SiPixelRecHitQuality::QualWordType 
PixelCPEBase::rawQualityWord() const
{
  SiPixelRecHitQuality::QualWordType qualWord(0);
  
  SiPixelRecHitQuality::thePacking.setProbabilityXY ( probabilityXY() ,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setProbabilityQ  ( probabilityQ_ , 
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setQBin          ( (int)qBin_, 
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setIsOnEdge      ( isOnEdge_,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setHasBadPixels  ( hasBadPixels_,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setSpansTwoROCs  ( spansTwoROCs_,
                                                      qualWord );
  
  SiPixelRecHitQuality::thePacking.setHasFilledProb ( hasFilledProb_,
                                                      qualWord );
  
  return qualWord;
}
