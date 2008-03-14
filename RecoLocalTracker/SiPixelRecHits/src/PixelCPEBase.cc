// Move geomCorrection to the concrete class. d.k. 06/06.
// Change drift direction. d.k. 06/06

// G. Giurgiu (ggiurgiu@pha.jhu.edu), 12/01/06, implemented the function: 
// computeAnglesFromDetPosition(const SiPixelCluster & cl, 
//			        const GeomDetUnit    & det ) const
//                                    09/09/07, replaced assert statements with throw cms::Exception 
//                                              and fix an invalid pointer check in setTheDet function 
//                                    09/21/07, implement caching of Lorentz drift direction


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

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
PixelCPEBase::PixelCPEBase(edm::ParameterSet const & conf, const MagneticField *mag) 
  : nRecHitsTotal_(0), nRecHitsUsedEdge_(0), theDet(0),
    cotAlphaFromCluster_(-99999.0), cotBetaFromCluster_(-99999.0),
    probabilityX_(-99999.0), probabilityY_(-99999.0), qBin_(-99999.0)
{
  //--- Lorentz angle tangent per Tesla
  theTanLorentzAnglePerTesla =
    conf.getParameter<double>("TanLorentzAnglePerTesla");

  //--- Algorithm's verbosity
  theVerboseLevel = 
    conf.getUntrackedParameter<int>("VerboseLevel",0);

  //-- Magnetic Field
  magfield_ = mag;

  //-- Switch on/off E.B 
  alpha2Order = conf.getParameter<bool>("Alpha2Order");
}

//-----------------------------------------------------------------------------
//  One function to cache the variables common for one DetUnit.
//-----------------------------------------------------------------------------
void
PixelCPEBase::setTheDet( const GeomDetUnit & det ) const 
{
  if ( theDet == &det )
    return;       // we have already seen this det unit
  
  //--- This is a new det unit, so cache it
  theDet = dynamic_cast<const PixelGeomDetUnit*>( &det );

  if ( !theDet ) 
    {
      // &&& Fatal error!  TO DO: throw an exception!
      
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
  theTopol
    = dynamic_cast<const RectangularPixelTopology*>( & (theDet->specificTopology()) );

  //---- The geometrical description of one module/plaquette
  theNumOfRow = theTopol->nrows();      // rows in x
  theNumOfCol = theTopol->ncolumns();   // cols in y
  std::pair<float,float> pitchxy = theTopol->pitch();
  thePitchX = pitchxy.first;            // pitch along x
  thePitchY = pitchxy.second;           // pitch along y

  //--- Find the offset
  MeasurementPoint  offset = 
    theTopol->measurementPosition( LocalPoint(0., 0.) );  
  theOffsetX = offset.x();
  theOffsetY = offset.y();

  //--- Find if the E field is flipped: i.e. whether it points
  //--- from the beam, or towards the beam.  (The voltages are
  //--- applied backwards on every other module in barrel and
  //--- blade in forward.)
  theSign = isFlipped() ? -1 : 1;
  
  //--- The Lorentz shift.
  theLShiftX = lorentzShiftX();
  theLShiftY = lorentzShiftY();
  
  // testing 
  if(thePart == GeomDetEnumerators::PixelBarrel) {
    //cout<<" lorentz shift "<<theLShiftX<<" "<<theLShiftY<<endl;
    theLShiftY=0.;
  }

  if (theVerboseLevel > 1) 
    {
      LogDebug("PixelCPEBase") << "***** PIXEL LAYOUT *****" 
			       << " thePart = " << thePart
			       << " theThickness = " << theThickness
			       << " thePitchX  = " << thePitchX 
			       << " thePitchY  = " << thePitchY 
			       << " theOffsetX = " << theOffsetX 
			       << " theOffsetY = " << theOffsetY 
			       << " theLShiftX  = " << theLShiftX;
    }
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
  LocalPoint lp = theTopol->localPosition( MeasurementPoint(xcenter, ycenter) );
  //float lp_mod = sqrt( lp.x()*lp.x() + lp.y()*lp.y() + lp.z()*lp.z() );

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

  // calculate angles
  alpha_ = atan2( gv_dot_gvz, gv_dot_gvx );
  beta_  = atan2( gv_dot_gvz, gv_dot_gvy );

  // calculate cotalpha and cotbeta
  //   cotalpha_ = 1.0/tan(alpha_);
  //   cotbeta_  = 1.0/tan(beta_ );
  // or like this
  cotalpha_ = gv_dot_gvx / gv_dot_gvz;
  cotbeta_  = gv_dot_gvy / gv_dot_gvz;

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
  LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
  
  // &&& Or, maybe we need to move to the local frame ???
  //  LocalVector localDir( theDet->toLocal(theState.globalDirection()));
  //thePart = theDet->type().part();
  
  float locx = localDir.x();
  float locy = localDir.y();
  float locz = localDir.z();
  
  alpha_ = acos(locx/sqrt(locx*locx+locz*locz));
  if ( isFlipped() )                    // &&& check for FPIX !!!
    alpha_ = PI - alpha_ ;
  
  beta_ = acos(locy/sqrt(locy*locy+locz*locz));

  // &&& In the above, why not use atan2() ?
  
  cotalpha_ = localDir.x()/localDir.z();
  cotbeta_  = localDir.y()/localDir.z();

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
//-----------------------------------------------------------------------------
LocalPoint
PixelCPEBase::localPosition( const SiPixelCluster& cluster, 
			     const GeomDetUnit & det) const {
  setTheDet( det );
  
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
  return theTopol->measurementPosition(lp);
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
  return theTopol->measurementError( lp, le );
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
  float tmp1 = theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
  float tmp2 = theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
  //cout << " 1: " << tmp1 << " 2: " << tmp2 << endl;
  if ( tmp2<tmp1 ) return true;
  else return false;    
}

//-----------------------------------------------------------------------------
// HALF OF the Lorentz shift (so for the full shift multiply by 2), and
// in the units of pitch.  (So note these are neither local nor measurement
// units!)
//-----------------------------------------------------------------------------
float PixelCPEBase::lorentzShiftX() const 
{

  LocalVector dir;
  
  Param & p = const_cast<PixelCPEBase*>(this)->m_Params[ theDet->geographicalId().rawId() ];
  if ( p.topology ) 
    {
      //cout << "--------------- old ----------------------" << endl;
      //cout << "p.topology = " << p.topology << endl;
      dir = p.drift;
      //cout << "same direction: dir = " << dir << endl;
    }
  else 
    {
      //cout << "--------------- new ----------------------" << endl;
      //cout << "p.topology = " << p.topology << endl;
      p.topology = (RectangularPixelTopology*)( & ( theDet->specificTopology() ) );    
      p.drift = driftDirection(magfield_->inTesla(theDet->surface().position()) );
      dir = p.drift;
      //cout << "p.topology = " << p.topology << endl;
      //cout << "new direction: dir = " << dir << endl;

    }

  //LocalVector dir = driftDirection(magfield_->inTesla(theDet->surface().position()) );
  
  // max shift in cm 
  float xdrift = dir.x()/dir.z() * theThickness;  
  // express the shift in units of pitch, 
  // divide by 2 to get the average correction
  float lshift = xdrift / thePitchX / 2.; 
  
  //cout << "Lorentz Drift = " << lshift << endl;
  //cout << "X Drift = " << dir.x() << endl;
  //cout << "Z Drift = " << dir.z() << endl;
  
  return lshift;  
  

}

float PixelCPEBase::lorentzShiftY() const 
{
  
  LocalVector dir;
  
  Param & p = const_cast<PixelCPEBase*>(this)->m_Params[ theDet->geographicalId().rawId() ];
  if ( p.topology ) 
    {
      //cout << "--------------- old y ----------------------" << endl;
      //cout << "p.topology y = " << p.topology << endl;
      dir = p.drift;
      //cout << "same direction y: dir = " << dir << endl;
    }
  else 
    {
      //cout << "--------------- new y ----------------------" << endl;
      //cout << "p.topology y = " << p.topology << endl;
      p.topology = (RectangularPixelTopology*)( & ( theDet->specificTopology() ) );    
      p.drift = driftDirection(magfield_->inTesla(theDet->surface().position()) );
      dir = p.drift;
      //cout << "p.topology y = " << p.topology << endl;
      //cout << "new direction y: dir = " << dir << endl;

    }

  //LocalVector dir = driftDirection(magfield_->inTesla(theDet->surface().position()) );
  
  float ydrift = dir.y()/dir.z() * theThickness;
  float lshift = ydrift / thePitchY / 2.;
  return lshift; 
  

}

//-----------------------------------------------------------------------------
//  Sum the pixels in the first and the last row, and the total.  Returns
//  a vector of three elements with q_first, q_last and q_total.
//  &&& Really need a simpler & cleaner way, this is very confusing...
//-----------------------------------------------------------------------------
void PixelCPEBase::xCharge(const vector<SiPixelCluster::Pixel>& pixelsVec, 
			   const int& imin, const int& imax,
			   float& q1, float& q2) const {
  //calculate charge in the first and last pixel in y
  // and the total cluster charge
  q1 = 0.0; 
  q2 = 0.0;
  float qm = 0.0;
  int isize = pixelsVec.size();
  for (int i=0;  i<isize; ++i) {
    if ( int(pixelsVec[i].x) == imin )
      q1 += pixelsVec[i].adc;
    else if ( int(pixelsVec[i].x) == imax) 
      q2 += pixelsVec[i].adc;
    else 
      qm += pixelsVec[i].adc;
  }
  return;
} 
//-----------------------------------------------------------------------------
//  Sum the pixels in the first and the last column, and the total.  Returns
//  a vector of three elements with q_first, q_last and q_total.
//  &&& Really need a simpler & cleaner way, this is very confusing...
//-----------------------------------------------------------------------------
void PixelCPEBase::yCharge(const vector<SiPixelCluster::Pixel>& pixelsVec,
			   const int& imin, const int& imax,
			   float& q1, float& q2) const {
  
  //calculate charge in the first and last pixel in y
  // and the inner cluster charge
  q1 = 0;
  q2 = 0;
  float qm=0;
  int isize = pixelsVec.size();
  for (int i=0;  i<isize; ++i) {
    if ( int(pixelsVec[i].y) == imin) 
      q1 += pixelsVec[i].adc;
    else if ( int(pixelsVec[i].y) == imax) 
      q2 += pixelsVec[i].adc;
    //else if (pixelsVec[i].y < ymax && pixelsVec[i].y > ymin ) 
    else  
      qm += pixelsVec[i].adc;
  }
  return;
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
  
  float alpha2;
  if (alpha2Order) {
      alpha2 = theTanLorentzAnglePerTesla*theTanLorentzAnglePerTesla;
  } else {
    alpha2 = 0.0;
  }
  
  // &&& dir_x should have a "-" and dir_y a "+"
  float dir_x =  ( theTanLorentzAnglePerTesla * Bfield.y() + alpha2* Bfield.z()* Bfield.x() );
  float dir_y = -( theTanLorentzAnglePerTesla * Bfield.x() - alpha2* Bfield.z()* Bfield.y() );
  float dir_z = -( 1 + alpha2* Bfield.z()*Bfield.z() );
  float scale = (1 + alpha2* Bfield.z()*Bfield.z() );
  LocalVector theDriftDirection = LocalVector(dir_x/scale, dir_y/scale, dir_z/scale );
   
  if ( theVerboseLevel > 9 ) 
      LogDebug("PixelCPEBase") << " The drift direction in local coordinate is " 
			       << theDriftDirection    ;
  
  return theDriftDirection;
}

//-----------------------------------------------------------------------------
//  One-shot computation of the driftDirection and both lorentz shifts
//-----------------------------------------------------------------------------
void
PixelCPEBase::computeLorentzShifts() const 
{
  Frame detFrame(theDet->surface().position(), theDet->surface().rotation());
  GlobalVector global_Bfield = magfield_->inTesla( theDet->surface().position() );
  LocalVector  Bfield        = detFrame.toLocal(global_Bfield);
  
  double alpha2;
  if ( alpha2Order) {
    alpha2 = theTanLorentzAnglePerTesla * theTanLorentzAnglePerTesla;
  }
  else  {
    alpha2 = 0.0;
  }

  // **********************************************************************
  // Our convention is the following:
  // +x is defined by the direction of the Lorentz drift!
  // +z is defined by the direction of E field (so electrons always go into -z!)
  // +y is defined by +x and +z, and it turns out to be always opposite to the +B field.
  // **********************************************************************
      
  // Note correct signs for dir_x and dir_y!
  double dir_x = -( theTanLorentzAnglePerTesla * Bfield.y() + alpha2* Bfield.z()* Bfield.x() );
  double dir_y =  ( theTanLorentzAnglePerTesla * Bfield.x() - alpha2* Bfield.z()* Bfield.y() );
  double dir_z = -( 1                                       + alpha2* Bfield.z()* Bfield.z() );

  // &&& Why do we need to scale???
  //double scale = (1 + alpha2* Bfield.z()*Bfield.z() );
  double scale = fabs( dir_z );  // same as 1 + alpha2*Bfield.z()*Bfield.z()
  driftDirection_ = LocalVector(dir_x/scale, dir_y/scale, dir_z/scale );  // last is -1 !

  // Max shift (at the other side of the sensor) in cm 
  lorentzShiftInCmX_ = driftDirection_.x()/driftDirection_.z() * theThickness;  // &&& redundant
  // Express the shift in units of pitch, 
  lorentzShiftX_ = lorentzShiftInCmX_ / thePitchX ; 
   
  // Max shift (at the other side of the sensor) in cm 
  lorentzShiftInCmY_ = driftDirection_.y()/driftDirection_.z() * theThickness;  // &&& redundant
  // Express the shift in units of pitch, 
  lorentzShiftY_ = lorentzShiftInCmY_ / thePitchY;


  if ( theVerboseLevel > 9 ) {
    LogDebug("PixelCPEBase") << " The drift direction in local coordinate is " 
			     << driftDirection_    ;
    
    cout << "Lorentz Drift (in cm) along X = " << lorentzShiftInCmX_ << endl;
    cout << "Lorentz Drift (in cm) along Y = " << lorentzShiftInCmY_ << endl;
  }
}
