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
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEInitial.h"

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
PixelCPEInitial::PixelCPEInitial(edm::ParameterSet const & conf, const MagneticField *mag) 
  : PixelCPEBase(conf,mag)
{
}


//------------------------------------------------------------------
//  Public methods mandated by the base class.
//------------------------------------------------------------------


//------------------------------------------------------------------
//  localPosition() calls measurementPosition() and then converts it
//  to the LocalPoint.
//------------------------------------------------------------------
LocalPoint
PixelCPEInitial::localPosition(const SiPixelCluster& cluster, const GeomDetUnit & det) const
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
PixelCPEInitial::localError( const SiPixelCluster& cluster, const GeomDetUnit & det)const 
{
  setTheDet( det );
  int sizex = cluster.sizeX();
  int sizey = cluster.sizeY();
  bool edgex = (cluster.edgeHitX()) || (cluster.maxPixelRow() > theNumOfRow); 
  bool edgey = (cluster.edgeHitY()) || (cluster.maxPixelCol() > theNumOfCol); 
  //&&& testing...
  if (theVerboseLevel > 5) {
    LogDebug("PixelCPEInitial") <<
      "Sizex = " << sizex << 
      " Sizey = " << sizey << 
      " Edgex = " << edgex << 
      " Edgey = " << edgey ;
  }
  //if (sizex>0) return LocalError( sizex, 0, sizey );

  return LocalError( err2X(edgex, sizex), 0, err2Y(edgey, sizey) );
}



//------------------------------------------------------------------
//  Helper methods (protected)
//------------------------------------------------------------------




//-----------------------------------------------------------------------------
//  Error along X, squared, as parameterized by Vincenzo.
//-----------------------------------------------------------------------------
float 
PixelCPEInitial::err2X(bool& edgex, int& sizex) const
{
// Assign maximum error
  // if edge cluster the maximum error is assigned: Pitch/sqrt(12)=43mu
  //  float xerr = 0.0043; 
  float xerr = thePitchX/3.464;
  //
  // Pixels not at the edge: errors parameterized as function of the cluster size
  // V.Chiochia - 12/4/06
  //
  if (!edgex){
    //    if (fabs(thePitchX-0.010)<0.001){   // 100um pixel size
      if (thePart == GeomDetType::PixelBarrel) {
	if ( sizex == 1) xerr = 0.00115;      // Size = 1 -> Sigma = 11.5 um 
	else if ( sizex == 2) xerr = 0.0012;  // Size = 2 -> Sigma = 12 um      
	else if ( sizex == 3) xerr = 0.00088; // Size = 3 -> Sigma = 8.8 um
	else xerr = 0.0103;
      } else { //forward
	if ( sizex == 1) {
	  xerr = 0.0020;
	}  else if ( sizex == 2) {
	  xerr = 0.0020;
	  // xerr = (0.005351 - atan(fabs(theDetZ/theDetR)) * 0.003291);  
	} else {
	  xerr = 0.0020;
	  //xerr = (0.003094 - atan(fabs(theDetZ/theDetR)) * 0.001854);  
	}
      }
      //    }
//     }else if (fabs(thePitchX-0.015)<0.001){  // 150 um pixel size
//       if (thePart == GeomDetType::PixelBarrel) {
// 	if ( sizex == 1) xerr = 0.0014;     // 14um 
// 	else xerr = 0.0008;   // 8um      
//       } else { //forward
// 	if ( sizex == 1) 
// 	  xerr = (-0.00385 + atan(fabs(theDetZ/theDetR)) * 0.00407);
// 	else xerr = (0.00366 - atan(fabs(theDetZ/theDetR)) * 0.00209);  
//       }
//     }

  }
  return xerr*xerr;
}




//-----------------------------------------------------------------------------
//  Error along Y, squared, as parameterized by Vincenzo.
//-----------------------------------------------------------------------------
float 
PixelCPEInitial::err2Y(bool& edgey, int& sizey) const
{
// Assign maximum error
// if edge cluster the maximum error is assigned: Pitch/sqrt(12)=43mu
//  float yerr = 0.0043;
  float yerr = thePitchY/3.464; 
  if (!edgey){
    if (thePart == GeomDetType::PixelBarrel) { // Barrel
      if ( sizey == 1) {
	yerr = 0.00375;     // 37.5um 
      } else if ( sizey == 2) {
	yerr = 0.0023;   // 23 um      
      } else if ( sizey == 3) {
	yerr = 0.0025; // 25 um
      } else if ( sizey == 4) {
	yerr = 0.0025; // 25um
      } else if ( sizey == 5) {
	yerr = 0.0023; // 23um
      } else if ( sizey == 6) {
	yerr = 0.0023; // 23um
      } else if ( sizey == 7) {
	yerr = 0.0021; // 21um
      } else if ( sizey == 8) {
	yerr = 0.0021; // 21um
      } else if ( sizey == 9) {
	yerr = 0.0024; // 24um
      } else if ( sizey >= 10) {
	yerr = 0.0021; // 21um
      }
    } else { // Endcaps
      if ( sizey == 1)      yerr = 0.0021; // 21 um
      else if ( sizey >= 2) yerr = 0.00075;// 7.5 um
    }
  }
  return yerr*yerr;
}



//-----------------------------------------------------------------------------
//  Calculates the *corrected* position of the cluster.
//  &&& Probably generic enough for the base class.
//-----------------------------------------------------------------------------
float 
PixelCPEInitial::xpos(const SiPixelCluster& cluster) const
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
//  Calculates the *corrected* position of the cluster.
//  &&& Probably generic enough for the base class.
//-----------------------------------------------------------------------------
float 
PixelCPEInitial::ypos(const SiPixelCluster& cluster) const
{
  float ycluster = 0;
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  int size = cluster.sizeY();
  float baryc = cluster.y();
  // &&& Testing...
  //if (baryc > 0) return baryc;

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
//
//-----------------------------------------------------------------------------
float 
PixelCPEInitial::chargeWidthX() const
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
PixelCPEInitial::chargeWidthY() const
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
// &&& NOT USED.  Remove?
//-----------------------------------------------------------------------------
float 
PixelCPEInitial::chaWidth2X(const float& centerx) const
{
  float chargeWX = chargeWidthX() + theSign * geomCorrection() * centerx;
  if ( chargeWX > 1. || chargeWX <= 0. ) chargeWX=1.;
  return chargeWX;
}


