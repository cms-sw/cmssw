//#include "Utilities/Configuration/interface/Architecture.h"

#include "Geometry/TrackerSimAlgo/interface/PixelGeomDetUnit.h"
//#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/RectangularPixelTopology.h"

//#include "CommonDet/BasicDet/interface/Topology.h"
//#include "CommonDet/BasicDet/interface/Det.h"
//#include "CommonDet/BasicDet/interface/DetUnit.h"
//#include "CommonDet/BasicDet/interface/DetType.h"
//#include "Tracker/SiPixelDet/interface/PixelDetType.h"
//#include "Tracker/SiPixelDet/interface/PixelDigi.h"
//#include "Tracker/SiPixelDet/interface/PixelTopology.h"
//  #include "CommonDet/DetGeometry/interface/ActiveMediaShape.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/CPEFromDetPosition.h"
//#define DEBUG

#include <iostream>
using namespace std;

const float PI = 3.141593;
const float degsPerRad = 57.29578;


//-----------------------------------------------------------------------------
//  A fairly boring constructor.  All quantities are DetUnit-dependent, and
//  will be initialized in setTheDet().
//-----------------------------------------------------------------------------
CPEFromDetPosition::CPEFromDetPosition(edm::ParameterSet const & conf)
{
  //--- Lorentz angle tangent per Tesla
  theTanLorentzAnglePerTesla =
    conf.getParameter<double>("TanLorentzAnglePerTesla");

  //--- Algorithm's verbosity
  theVerboseLevel = 
    conf.getParameter<int>("VerboseLevel");
}


//-----------------------------------------------------------------------------
//  One function to cache the variables common for one DetUnit.
//-----------------------------------------------------------------------------
void
CPEFromDetPosition::setTheDet( const GeomDetUnit & det )
{
  if ( theDet == &det )
    return;       // we have already seen this det unit

  //--- This is a new det unit, so cache it's values
  theDet = dynamic_cast<const PixelGeomDetUnit*>( &det );
  if (! theDet) {
    // &&& Fatal error!  TO DO: throw an exception!
    assert(0);
  }

  //--- theDet->type() returns a GeomDetType, which implements subDetector()
  thePart = theDet->type().subDetector();
  switch (thePart) {
  case GeomDetType::PixelBarrel:
    // A barrel!  A barrel!
    break;
  case GeomDetType::PixelEndcap:
    // A forward!  A forward!
    break;
  default:
    std::cout 
      << "CPEFromDetPosition:: a non-pixel detector type in here? Yuck!" 
      << std::endl;
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
  theLShift = lorentzShift();

#ifdef DEBUG
  cout << "***** PIXEL LAYOUT *****" << endl;
  cout << " theThickness = " << theThickness << endl;
  cout << " thePitchX  = " << thePitchX << endl;
  cout << " thePitchY  = " << thePitchY << endl;
  cout << " theOffsetX  = " << theOffsetX << endl;
  cout << " theOffsetY  = " << theOffsetY << endl;
#endif
}

MeasurementError  
CPEFromDetPosition::measurementError( const SiPixelCluster& cluster, const GeomDetUnit & det) 
{
  LocalPoint lp( localPosition(cluster, det) );
  LocalError le( localError(   cluster, det) );
  return theTopol->measurementError( lp, le );
}

LocalError  
CPEFromDetPosition::localError( const SiPixelCluster& cluster, const GeomDetUnit & det)
{
  setTheDet( det );
  int sizex = cluster.sizeX();
  int sizey = cluster.sizeY();
  bool edgex = (cluster.edgeHitX()) || (cluster.maxPixelRow()> theNumOfRow); 
  bool edgey = (cluster.edgeHitY()) || (cluster.maxPixelCol() > theNumOfCol); 
  return LocalError( err2X(edgex, sizex), 0, err2Y(edgey, sizey) );
}

MeasurementPoint 
CPEFromDetPosition::measurementPosition( const SiPixelCluster& cluster, const GeomDetUnit & det) 
{
  
  return MeasurementPoint( xpos(cluster)-theLShift, 
  			   ypos(cluster));
}

LocalPoint
CPEFromDetPosition::localPosition(const SiPixelCluster& cluster, const GeomDetUnit & det) 
{
  return theTopol->localPosition(measurementPosition(cluster, det)); 
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::err2X(bool& edgex, int& sizex) const
{
// Assign maximum error
  // if edge cluster the maximum error is assigned: Pitch/sqrt(12)=43mu
  //  float xerr = 0.0043; 
  float xerr = thePitchX/3.464; 
  if (!edgex){
    if (fabs(thePitchX-0.010)<0.001){   // 100um pixel size
      if (thePart == GeomDetType::PixelBarrel) {
	if ( sizex == 1) xerr = 0.0010;     // 10um 
	else if ( sizex == 2) xerr = 0.0009;   // 9um      
	else xerr = 0.00055;   // 5.5um      
      } else { //forward
	if ( sizex == 1) {
	  xerr = 0.001;
	}  else if ( sizex == 2) {
	  xerr = (0.005351 - atan(fabs(theDetZ/theDetR)) * 0.003291);  
	} else {
	  xerr = (0.003094 - atan(fabs(theDetZ/theDetR)) * 0.001854);  
	}
      }
    }else if (fabs(thePitchX-0.015)<0.001){  // 150 um pixel size
      if (thePart == GeomDetType::PixelBarrel) {
	if ( sizex == 1) xerr = 0.0014;     // 14um 
	else xerr = 0.0008;   // 8um      
      } else { //forward
	if ( sizex == 1) 
	  xerr = (-0.00385 + atan(fabs(theDetZ/theDetR)) * 0.00407);
	else xerr = (0.00366 - atan(fabs(theDetZ/theDetR)) * 0.00209);  
      }
    }
  }
  return xerr*xerr;
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::err2Y(bool& edgey, int& sizey) const
{
// Assign maximum error
// if edge cluster the maximum error is assigned: Pitch/sqrt(12)=43mu
//  float yerr = 0.0043;
  float yerr = thePitchY/3.464; 
  if (!edgey){
    if (thePart == GeomDetType::PixelBarrel) {
      if ( sizey == 1) {
	yerr = 0.0030;     // 31um 
      } else if ( sizey == 2) {
	yerr = 0.0021;   // 18um      
      } else if ( sizey == 3) {
	yerr = 0.0020; // 20um 
      } else {
	yerr = 0.0017;   // 17um
      }      
    } else { //forward
      if ( sizey == 1) yerr = 0.0022; // 22um
      else yerr = 0.0008;  // 7um
    }
  }
  return yerr*yerr;
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::xpos(const SiPixelCluster& cluster) const
{
  float xcluster = 0;
  int size = cluster.sizeX();
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  float baryc = cluster.x();
  if (size == 1) {
    // the middle of only one pixel is equivalent to the baryc.
    xcluster = baryc;
  } else {

    //calculate center
    float xmin = float(cluster.minPixelRow()) + 0.5;
    float xmax = float(cluster.maxPixelRow()) + 0.5;
    float xcenter = ( xmin + xmax ) / 2;

    vector<float> xChargeVec = xCharge(pixelsVec, xmin, xmax); 
    float q1 = xChargeVec[0];
    float q2 = xChargeVec[1];
    float chargeWX = chargeWidthX() + theSign * geomCorrection() * xcenter;
    float effchargeWX = fabs(chargeWX) - (float(size)-2);

    // truncated charge width only if it greather than the cluster size
    if ( fabs(effchargeWX) > 2 ) effchargeWX = 1;

    xcluster = xcenter + (q2-q1) * effchargeWX / (q1+q2) / 2.;


    float alpha = estimatedAlphaForBarrel(xcenter);
    if (alpha < 1.53) {
      float etashift=0;
      float charatio = q1/(q1+q2);
      etashift = theEtaFunc.xEtaShift(size, thePitchX, 
				      charatio, alpha);
      xcluster = xcluster - etashift;
    }
  }    
  return xcluster;
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::ypos(const SiPixelCluster& cluster) const
{
  float ycluster = 0;
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  int size = cluster.sizeY();
  float baryc = cluster.y();
  if (size == 1) {
    ycluster = baryc;
  } else if (size < 4) {

    // Calculate center
    float ymin = float(cluster.minPixelCol()) + 0.5;
    float ymax = float(cluster.maxPixelCol()) + 0.5;
    float ycenter = ( ymin + ymax ) / 2;

    //calculate charge width
    float chargeWY = chargeWidthY() + geomCorrection() * ycenter;
    float effchargeWY = fabs(chargeWY) - (float(size)-2);

    // truncate charge width when it is > 2
    if ( (effchargeWY < 0) || (effchargeWY > 1.) ) effchargeWY = 1;

    //calculate charge of first, last and inner pixels of cluster
    vector<float> yChargeVec = yCharge(pixelsVec, ymin, ymax);
    float q1 = yChargeVec[0];
    float q2 = yChargeVec[1];
    // float qm = yChargeVec[2];
    // float charatio = q1/(q1+q2);

    ycluster = ycenter + (q2-q1) * effchargeWY / (q1+q2) / 2.;

  } else {    //  Use always the edge method
    // Calculate center
    float ymin = float(cluster.minPixelCol()) + 0.5;
    float ymax = float(cluster.maxPixelCol()) + 0.5;
    float ycenter = ( ymin + ymax ) / 2;
    
    //calculate charge of first, last and inner pixels of cluster
    vector<float> yChargeVec = yCharge(pixelsVec, ymin, ymax);
    float q1 = yChargeVec[0];
    float q2 = yChargeVec[1];
    
    float shift = (q2 - q1) / (q2 + q1) / 2.; // Single edge 

    ycluster = ycenter + shift;

  }
  return ycluster;
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
//
// Plan: ignore it for the moment
//-----------------------------------------------------------------------------
bool 
CPEFromDetPosition::isFlipped() const
{
// &&& Not sure what the need is -- ask Danek.
//   float tmp1 = theDet->toGlobal( Local3DPoint(0.,0.,0.) ).perp();
//   float tmp2 = theDet->toGlobal( Local3DPoint(0.,0.,1.) ).perp();
//   if ( tmp2<tmp1 ) return true;
//   else return false;
  return false;    // &&& TEMPORARY HACK
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::chargeWidthX() const
{ 
  float chargeW = 0;
  float lorentzWidth = 2 * theLShift;
  if (thePart == GeomDetType::PixelBarrel) {
    // Redefine the charge width to include the offset
    chargeW = lorentzWidth - theSign * geomCorrection() * theOffsetX;
  } else { // forward
    chargeW = fabs(lorentzWidth) + 
      theThickness * fabs(theDetR/theDetZ) / thePitchX;
  }
  return chargeW;
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::chargeWidthY() const
{
  float chargeW = 0;  
  if (thePart == GeomDetType::PixelBarrel) {
    chargeW = theThickness * fabs(theDetZ/theDetR) / thePitchY;
    chargeW -= (geomCorrection() * theOffsetY);
  } else { //forward
    // Width comes from geometry only, fixed by the tilt angle
   chargeW = theThickness * tan(20./degsPerRad) / thePitchY; 
  }
  return chargeW;
}


//-----------------------------------------------------------------------------
// From Danek: "geomCorrection() is sort of second order effect, ignore it for 
// the moment. I have to to derive it again and understand better what it means."
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::geomCorrection() const
{ 
  //@@ the geometrical correction are calculated only
  //@@ for the barrel part (am I right?)  &&& ??????????????????
  if (thePart == GeomDetType::PixelEndcap) return 0;
  else return theThickness / theDetR;
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::lorentzShift() const
{
  // Implement only the x direction shift for now (OK for barrel)
  // &&& TEMPORARY LocalVector dir = theDet->driftDirection( LocalPoint(0,0));
  LocalVector dir(0,0,0);  // &&& TEMPORARY HACK

  // max shift in cm 
  float xdrift = dir.x()/dir.z() * theThickness;  
  // express the shift in units of pitch, 
  // divide by 2 to get the average correction
  float lshift = xdrift / thePitchX / 2.; 
#ifdef DEBUG
  cout << "Lorentz Drift = " << xdrift << endl;
  cout << "X Drift = " << dir.x() << endl;
  cout << "Z Drift = " << dir.z() << endl;
#endif 
  return lshift;  
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::chaWidth2X(const float& centerx) const
{
  float chargeWX = chargeWidthX() + theSign * geomCorrection() * centerx;
  if ( chargeWX > 1. || chargeWX <= 0. ) chargeWX=1.;
  return chargeWX;
}



//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::estimatedAlphaForBarrel(const float& centerx) const
{
  float tanalpha = theSign*(centerx-theOffsetX)/theDetR*thePitchX;
  return PI/2-atan(tanalpha);
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
vector<float> 
CPEFromDetPosition::xCharge(const vector<SiPixelCluster::Pixel>& pixelsVec, 
				    const float& xmin, const float& xmax)const
{
  vector<float> charge; 

  //calculate charge in the first and last pixel in y
  // and the total cluster charge
  float q1 = 0, q2 = 0, qm=0;
  int isize = pixelsVec.size();
  for (int i = 0;  i < isize; ++i) {
    if (pixelsVec[i].x == xmin) q1 += pixelsVec[i].adc;
    else if (pixelsVec[i].x == xmax) q2 += pixelsVec[i].adc;
    else qm += pixelsVec[i].adc;
  }
  charge.clear();
  charge.push_back(q1); 
  charge.push_back(q2); 
  charge.push_back(qm);
  return charge;
} 


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
vector<float> 
CPEFromDetPosition::yCharge(const vector<SiPixelCluster::Pixel>& pixelsVec,
				    const float& ymin, const float& ymax)const
{
  vector<float> charge; 

  //calculate charge in the first and last pixel in y
  // and the inner cluster charge
  float q1 = 0, q2 = 0, qm=0;
  int isize = pixelsVec.size();
  for (int i = 0;  i < isize; ++i) {
    if (pixelsVec[i].y == ymin) q1 += pixelsVec[i].adc;
    else if (pixelsVec[i].y == ymax) q2 += pixelsVec[i].adc;
    else if (pixelsVec[i].y < ymax && 
	     pixelsVec[i].y > ymin ) qm += pixelsVec[i].adc;
  }
  charge.clear();
  charge.push_back(q1); 
  charge.push_back(q2); 
  charge.push_back(qm);

  return charge;
} 




//-----------------------------------------------------------------------------
//  Drift direction.
//  NB: it's correct only for the barrel!  &&& Need to fix it for the forward.
//  Assumption: setTheDet() has been called already.
//-----------------------------------------------------------------------------
LocalVector 
CPEFromDetPosition::driftDirection( GlobalVector bfield )
{
  Frame detFrame(theDet->surface().position(), theDet->surface().rotation());
  LocalVector Bfield = detFrame.toLocal(bfield);


  //  if    (DetId(detID).subdetId()==  PixelSubdetector::PixelBarrel){
  float dir_x =  theTanLorentzAnglePerTesla * Bfield.y();
  float dir_y = -theTanLorentzAnglePerTesla * Bfield.x();
  float dir_z = 1.; // E field always in z direction
  LocalVector theDriftDirection = LocalVector(dir_x,dir_y,dir_z);

  if (theVerboseLevel > 0) {
    cout << " The drift direction in local coordinate is " 
  	 << theDriftDirection    << endl;
  }

  return theDriftDirection;
  //  }
}




