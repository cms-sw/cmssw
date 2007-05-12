
// Move geomCorrection to the concrete class. d.k. 06/06.
// Change drift direction. d.k. 06/06

// G. Giurgiu (ggiurgiu@pha.jhu.edu), 12/01/06, implemented the function: 
// computeAnglesFromDetPosition(const SiPixelCluster & cl, 
//			        const GeomDetUnit    & det ) const

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
  if (! theDet) 
    {
      // &&& Fatal error!  TO DO: throw an exception!
      assert(0);
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
      LogDebug("PixelCPEBase") 
	<< "PixelCPEBase:: a non-pixel detector type in here?" ;
      //  &&& Should throw an exception here!
      assert(0);
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
  if (! theDet) 
    {
      // &&& Fatal error!  TO DO: throw an exception!
      cout << "---------------------------------------------- Not a pixel detector !!!!!!!!!!!!!!" << endl;
      assert(0);
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
  cotalpha_ = 1.0/tan(alpha_);
  cotbeta_  = 1.0/tan(beta_ );
  // or like this
  //cotalpha_ = gv_dot_gvx / gv_dot_gvz;
  //cotbeta_  = gv_dot_gvy / gv_dot_gvz;

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
//-----------------------------------------------------------------------------
LocalPoint
PixelCPEBase::localPosition( const SiPixelCluster& cluster, 
			     const GeomDetUnit & det) const 
{
  setTheDet( det );
  
#ifdef CORRECT_FOR_BIG_PIXELS
  MeasurementPoint ssss( xpos(cluster),ypos(cluster) );
  LocalPoint lp = theTopol->localPosition(ssss);
  if ( alpha2Order) 
    {
      float lxshift = theLShiftX * thePitchX;  // shift in cm
      float lyshift = theLShiftY*thePitchY;
      if ( thePart == GeomDetEnumerators::PixelBarrel )
	{
	  LocalPoint cdfsfs(lp.x()-lxshift, lp.y());
	  return cdfsfs;
	} 
      else 
	{ //forward
	  LocalPoint cdfsfs(lp.x()-lxshift, lp.y()-lyshift);
	  return cdfsfs;
	}
    }
  else 
    {
      float lxshift = theLShiftX * thePitchX;  // shift in cm
      LocalPoint cdfsfs(lp.x()-lxshift, lp.y() );
      return cdfsfs;
    }
  
#else
  MeasurementPoint ssss = measurementPosition(cluster, det);
  LocalPoint cdfsfs = theTopol->localPosition(ssss);
  return cdfsfs;
#endif

 // return cdfsfs;
}

//-----------------------------------------------------------------------------
//  Takes the cluster, calculates xpos() and ypos(), applies the Lorentz
//  shift, and then makes a MeasurementPoint.  This could really be
//  folded back into the localPosition().
//-----------------------------------------------------------------------------
MeasurementPoint 
PixelCPEBase::measurementPosition( const SiPixelCluster& cluster, 
				   const GeomDetUnit & det) const 
{
  if (theVerboseLevel > 9) 
    {
      LogDebug("PixelCPEBase") <<
	"X-pos = " << xpos(cluster) << 
	" Y-pos = " << ypos(cluster) << 
	" Lshf = " << theLShiftX ;
    }

  // Fix to take into account the large pixels
#ifdef CORRECT_FOR_BIG_PIXELS
  // correct the measurement for Lorentz shift
  if ( alpha2Order) 
    {
      float xPos = xpos(cluster); // x position in the measurement frame
      float yPos = ypos(cluster);
      float lxshift = theLShiftX; // nominal lorentz shift
      float lyshift = theLShiftY;
      if ( RectangularPixelTopology::isItBigPixelInX(int(xPos)) ) // if big
	lxshift = theLShiftX/2.0;  // reduce the shift
     if ( RectangularPixelTopology::isItBigPixelInY(int(yPos)) ) // if big
       lyshift = theLShiftY/2.0;  // reduce the shift
     
     if ( thePart == GeomDetEnumerators::PixelBarrel ) 
       {
	 return MeasurementPoint( xpos(cluster)-lxshift,ypos(cluster));
       } 
     else 
       { //forward
	 return MeasurementPoint( xpos(cluster)-lxshift,ypos(cluster)-lyshift);
       }
    }
  else 
    {
      float xPos = xpos(cluster); // x position in the measurement frame
      float lshift = theLShiftX; // nominal lorentz shift
      if ( RectangularPixelTopology::isItBigPixelInX(int(xPos)) ) // if big
	lshift = theLShiftX/2.0;  // reduce the shift
      return MeasurementPoint( xpos(cluster)-lshift,ypos(cluster));
    }
#else
  if ( alpha2Order) 
    {
      if ( thePart == GeomDetEnumerators::PixelBarrel) 
	{
	  return MeasurementPoint( xpos(cluster)-theLShiftX,ypos(cluster) );
	} 
      else 
	{ //forward
	  return MeasurementPoint( xpos(cluster)-theLShiftX, ypos(cluster)-theLShiftY);
	}
    }
  else 
    {
      return MeasurementPoint( xpos(cluster)-theLShiftX,
			       ypos(cluster) );
    }
  
  // skip the correction, do it only for the local position
  // in this mode the measurements are NOT corrected for the Lorentz shift
  //return MeasurementPoint( xpos(cluster),ypos(cluster));
#endif

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
// Lorentz shift. For the moment only in X direction (barrel & endcaps)
// For the forward the y componenet might have to be added.
//-----------------------------------------------------------------------------
float PixelCPEBase::lorentzShiftX() const 
{
  LocalVector dir = driftDirection(magfield_->inTesla(theDet->surface().position()) );
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
  LocalVector dir = driftDirection(magfield_->inTesla(theDet->surface().position()) );
  float ydrift = dir.y()/dir.z() * theThickness;
  float lshift = ydrift / thePitchY / 2.;
  return lshift;
}

//-----------------------------------------------------------------------------
//  Sum the pixels in the first and the last row, and the total.  Returns
//  a vector of three elements with q_first, q_last and q_total.
//  &&& Really need a simpler & cleaner way, this is very confusing...
//-----------------------------------------------------------------------------
vector<float> 
PixelCPEBase::xCharge(const vector<SiPixelCluster::Pixel>& pixelsVec, 
		      const float& xmin, const float& xmax) const
{
  vector<float> charge; 
  
  //calculate charge in the first and last pixel in y
  // and the total cluster charge
  float q1 = 0.0; 
  float q2 = 0.0;
  float qm = 0.0;
  int isize = pixelsVec.size();
  for (int i=0;  i<isize; ++i) 
    {
      if ( pixelsVec[i].x == xmin )
	q1 += pixelsVec[i].adc;
      else if (pixelsVec[i].x == xmax) 
	q2 += pixelsVec[i].adc;
      else 
	qm += pixelsVec[i].adc;
    }
  charge.clear();
  charge.push_back(q1); 
  charge.push_back(q2); 
  charge.push_back(qm);
  return charge;
} 

//-----------------------------------------------------------------------------
//  Sum the pixels in the first and the last column, and the total.  Returns
//  a vector of three elements with q_first, q_last and q_total.
//  &&& Really need a simpler & cleaner way, this is very confusing...
//-----------------------------------------------------------------------------
vector<float> 
PixelCPEBase::yCharge(const vector<SiPixelCluster::Pixel>& pixelsVec,
		      const float& ymin, const float& ymax) const
{
  vector<float> charge; 
  
  //calculate charge in the first and last pixel in y
  // and the inner cluster charge
  float q1 = 0, q2 = 0, qm=0;
  int isize = pixelsVec.size();
  for (int i=0;  i<isize; ++i) 
    {
      if (pixelsVec[i].y == ymin) 
	q1 += pixelsVec[i].adc;
      else if (pixelsVec[i].y == ymax) 
	q2 += pixelsVec[i].adc;
      else if (pixelsVec[i].y < ymax && pixelsVec[i].y > ymin ) 
	qm += pixelsVec[i].adc;
    }
  charge.clear();
  charge.push_back(q1); 
  charge.push_back(q2); 
  charge.push_back(qm);
  
  return charge;
} 

//-----------------------------------------------------------------------------
//  Drift direction.
//  Works OK for barrel and forward.
//  The formulas used for dir_x,y,z have to be exactly the same as the ones
//  used in the digitizer (SiPixelDigitizerAlgorithm.cc).
//  Assumption: setTheDet() has been called already.
//-----------------------------------------------------------------------------
LocalVector 
PixelCPEBase::driftDirection( GlobalVector bfield ) const 
{
  Frame detFrame(theDet->surface().position(), theDet->surface().rotation());
  LocalVector Bfield = detFrame.toLocal(bfield);
  
  float alpha2;
  if ( alpha2Order) 
    {
      alpha2 = theTanLorentzAnglePerTesla*theTanLorentzAnglePerTesla;
    }
  else 
    {
      alpha2 = 0.0;
    }
  
  float dir_x =  ( theTanLorentzAnglePerTesla * Bfield.y() + alpha2* Bfield.z()* Bfield.x() );
  float dir_y = -( theTanLorentzAnglePerTesla * Bfield.x() - alpha2* Bfield.z()* Bfield.y() );
  float dir_z = -( 1 + alpha2* Bfield.z()*Bfield.z() );
  float scale = (1 + alpha2* Bfield.z()*Bfield.z() );
  LocalVector theDriftDirection = LocalVector(dir_x/scale, dir_y/scale, dir_z/scale );
   
  // float dir_x =  theTanLorentzAnglePerTesla * Bfield.y();
  // float dir_y = -theTanLorentzAnglePerTesla * Bfield.x();
  // float dir_z = -1.; // E field always in z direction, so electrons go to -z.
  // LocalVector theDriftDirection = LocalVector(dir_x,dir_y,dir_z);
  
  if ( theVerboseLevel > 9 ) 
    {
      LogDebug("PixelCPEBase") << " The drift direction in local coordinate is " 
			       << theDriftDirection    ;
    }
  
  return theDriftDirection;
}




