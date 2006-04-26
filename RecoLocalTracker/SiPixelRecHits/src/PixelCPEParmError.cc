//#include "Utilities/Configuration/interface/Architecture.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
//#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

//#include "CommonDet/BasicDet/interface/Topology.h"
//#include "CommonDet/BasicDet/interface/Det.h"
//#include "CommonDet/BasicDet/interface/DetUnit.h"
//#include "CommonDet/BasicDet/interface/DetType.h"
//#include "Tracker/SiPixelDet/interface/PixelDetType.h"
//#include "Tracker/SiPixelDet/interface/PixelDigi.h"
//#include "Tracker/SiPixelDet/interface/PixelTopology.h"
//  #include "CommonDet/DetGeometry/interface/ActiveMediaShape.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEParmError.h"

//#define DEBUG

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

#include <iostream>
using namespace std;

const float PI = 3.141593;
const float degsPerRad = 57.29578;


//-----------------------------------------------------------------------------
//  A fairly boring constructor.  All quantities are DetUnit-dependent, and
//  will be initialized in setTheDet().
//-----------------------------------------------------------------------------
PixelCPEParmError::PixelCPEParmError(edm::ParameterSet const & conf, 
				     const MagneticField *mag) 
  : PixelCPEBase(conf,mag)
{
  pixelErrorParametrization_ = new PixelErrorParametrization(conf);
}

//-----------------------------------------------------------------------------
//  Clean up.
//-----------------------------------------------------------------------------
PixelCPEParmError::~PixelCPEParmError()
{
  delete pixelErrorParametrization_ ;
}


//------------------------------------------------------------------
//  Public methods mandated by the base class.
//------------------------------------------------------------------


//------------------------------------------------------------------
//  localPosition() calls measurementPosition() and then converts it
//  to the LocalPoint.
//------------------------------------------------------------------
LocalPoint
PixelCPEParmError::localPosition(const SiPixelCluster& cluster, const GeomDetUnit & det) const
{
  //return theTopol->localPosition(measurementPosition(cluster, det)); 
  setTheDet( det );
  MeasurementPoint ssss = measurementPosition(cluster, det);


  LocalPoint cdfsfs = theTopol->localPosition(ssss);
  return cdfsfs;
}


//------------------------------------------------------------------
//  localError() calls measurementError() after computing size and 
//  edge (flag) along x and y.
//------------------------------------------------------------------
LocalError  
PixelCPEParmError::localError( const SiPixelCluster& cluster, const GeomDetUnit & det)const 
{
  setTheDet( det );

  //--- Default is the maximum error used for edge clusters.
  float xerr = thePitchX / sqrt(12.);
  float yerr = thePitchY / sqrt(12.);

  //--- Are we near either of the edges?
  bool edgex = (cluster.edgeHitX()) || (cluster.maxPixelRow() > theNumOfRow); 
  bool edgey = (cluster.edgeHitY()) || (cluster.maxPixelCol() > theNumOfCol); 

  if (edgex && edgey) {
    //--- Both axes on the edge, no point in calling PixelErrorParameterization,
    //--- just return the max errors on both.
    // return LocalError(xerr*xerr, 0,yerr*yerr);
  }
  else {
    pair<float,float> errPair = 
      pixelErrorParametrization_->getError(thePart, 
					   cluster.sizeX(), cluster.sizeY(), 
					   alpha_         , beta_);
    if (!edgex) xerr = errPair.first;
    if (!edgey) yerr = errPair.second;
  }       

  if (theVerboseLevel > 5) {
    LogDebug("PixelCPEParmError") <<
      "Sizex = " << cluster.sizeX() << " Sizey = " << cluster.sizeY() << 
      " Edgex = " << edgex          << " Edgey = " << edgey << 
      " ErrX = " << xerr            << " ErrY  = " << yerr;
  }
  return LocalError(xerr*xerr, 0,yerr*yerr);
}



//------------------------------------------------------------------
//  Helper methods (protected)
//------------------------------------------------------------------







//-----------------------------------------------------------------------------
//  Calculates the *corrected* position of the cluster.
//  &&& Probably generic enough for the base class.
//-----------------------------------------------------------------------------
float 
PixelCPEParmError::xpos(const SiPixelCluster& cluster) const
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
    // &&& The following line from CPEFromDetPosition:
    // float chargeWX = chargeWidthX() + theSign * geomCorrection() * xcenter;
    float chargeWX = chargeWidthX();
    float effchargeWX = fabs(chargeWX) - (float(size)-2);

    // truncated charge width only if it greather than the cluster size
    if ( fabs(effchargeWX) > 2 ) effchargeWX = 1;

    xcluster = xcenter + (q2-q1) * effchargeWX / (q1+q2) / 2.;


    // &&& should go away there too:  float alpha = estimatedAlphaForBarrel(xcenter);
    if (alpha_ < 1.53) {
      float etashift=0;
      float charatio = q1/(q1+q2);
      etashift = theEtaFunc.xEtaShift(size, thePitchX, 
				      charatio, alpha_);
      xcluster = xcluster - etashift;
    }
  }    
  return xcluster;
}


//-----------------------------------------------------------------------------
//  Calculates the *corrected* position of the cluster.
//  &&& Probably generic enough for the base class.
//-----------------------------------------------------------------------------
float 
PixelCPEParmError::ypos(const SiPixelCluster& cluster) const
{
  float ycluster = 0;
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  int size = cluster.sizeY();
  float baryc = cluster.y();

  if (size == 1) {
    ycluster = baryc;
  } 
// &&& The size == 2,3 exists in FromDetPosition but not in FromTrackAngles:
//   else if (size < 4) {

//     // Calculate center
//     float ymin = float(cluster.minPixelCol()) + 0.5;
//     float ymax = float(cluster.maxPixelCol()) + 0.5;
//     float ycenter = ( ymin + ymax ) / 2;

//     //calculate charge width
//     float chargeWY = chargeWidthY() + geomCorrection() * ycenter;
//     float effchargeWY = fabs(chargeWY) - (float(size)-2);

//     // truncate charge width when it is > 2
//     if ( (effchargeWY < 0) || (effchargeWY > 1.) ) effchargeWY = 1;

//     //calculate charge of first, last and inner pixels of cluster
//     vector<float> yChargeVec = yCharge(pixelsVec, ymin, ymax);
//     float q1 = yChargeVec[0];
//     float q2 = yChargeVec[1];
//     // float qm = yChargeVec[2];
//     // float charatio = q1/(q1+q2);

//     ycluster = ycenter + (q2-q1) * effchargeWY / (q1+q2) / 2.;

//   } 
  else {    //  Use always the edge method
    
    float chargeWY = chargeWidthY();
    float effchargeWY = fabs(chargeWY) - (float(size)-2);
    // truncate charge width when it is > 2
    if ( (effchargeWY < 0) || (effchargeWY > 2) ) effchargeWY = 1;

    // Calculate center
    float ymin = float(cluster.minPixelCol()) + 0.5;
    float ymax = float(cluster.maxPixelCol()) + 0.5;
    float ycenter = ( ymin + ymax ) / 2;

    //calculate charge of first, last and inner pixels of cluster
    vector<float> yChargeVec = yCharge(pixelsVec, ymin, ymax);
    float q1 = yChargeVec[0];
    float q2 = yChargeVec[1];
    // float qm = yChargeVec[2];
    float charatio = q1/(q1+q2);

    // &&& FromDetPosition does not apply etashfit in y:
    // eta function for shallow tracks
    float etashift = theEtaFunc.yEtaShift(size, thePitchY, 
					  charatio, beta_);
    ycluster = ycenter + (q2-q1) * effchargeWY / (q1+q2) / 2.- etashift;

  } 

  return ycluster;
}



//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
PixelCPEParmError::chargeWidthX() const
{ 
// &&& Commented out: the version from FromDetPosition:
//   float chargeW = 0;
//   float lorentzWidth = 2 * theLShift;
//   if (thePart == GeomDetType::PixelBarrel) {
//     // Redefine the charge width to include the offset
//     chargeW = lorentzWidth - theSign * geomCorrection() * theOffsetX;
//   } else { // forward
//     chargeW = fabs(lorentzWidth) + 
//       theThickness * fabs(theDetR/theDetZ) / thePitchX;
//   }
//   return chargeW;

  float geomWidthX = theThickness * tan(PI/2 - alpha_)/thePitchX;
  if (thePart == GeomDetType::PixelBarrel){
    return (geomWidthX) + (2 * theLShift);
  } else {
    return fabs(geomWidthX) + (2 * fabs(theLShift)); 
  }
}


//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
float 
PixelCPEParmError::chargeWidthY() const
{
// &&& Commented out: the version from FromDetPosition:
//   float chargeW = 0;  
//   if (thePart == GeomDetType::PixelBarrel) {
//     chargeW = theThickness * fabs(theDetZ/theDetR) / thePitchY;
//     chargeW -= (geomCorrection() * theOffsetY);
//   } else { //forward
//     // Width comes from geometry only, fixed by the tilt angle
//    chargeW = theThickness * tan(20./degsPerRad) / thePitchY; 
//   }
//   return chargeW;

  float geomWidthY = theThickness * tan(PI/2 - beta_)/thePitchY;
  if (thePart == GeomDetType::PixelBarrel) {
    return geomWidthY;
  } else {
    return fabs(geomWidthY);
  }
}





