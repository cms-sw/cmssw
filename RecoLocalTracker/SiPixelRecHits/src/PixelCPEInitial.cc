// Move geomCorrection from the base class, modify it. 
// comment out etaCorrection. d.k. 06/06
// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEInitial.h"

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

#include <iostream>

//#define TDEBUG
using namespace std;

const float PI = 3.141593;
const float degsPerRad = 57.29578;

//-----------------------------------------------------------------------------
//  A fairly boring constructor.  All quantities are DetUnit-dependent, and
//  will be initialized in setTheDet().
//-----------------------------------------------------------------------------
PixelCPEInitial::PixelCPEInitial(edm::ParameterSet const & conf, const MagneticField *mag, const SiPixelLorentzAngle * lorentzAngle) 
  : PixelCPEBase(conf,mag,lorentzAngle)
{
}


//------------------------------------------------------------------
//  Public methods mandated by the base class.
//------------------------------------------------------------------

//------------------------------------------------------------------
//  localError() calls measurementError() after computing size and 
//  edge (flag) along x and y.
//------------------------------------------------------------------
LocalError  
PixelCPEInitial::localError( const SiPixelCluster& cluster, const GeomDetUnit & det)const 
{
  setTheDet( det, cluster );
  int sizex = cluster.sizeX();
  int sizey = cluster.sizeY();

  // Find edge clusters
  // Use edge methods from the Toplogy class
  int maxPixelCol = cluster.maxPixelCol();
  int maxPixelRow = cluster.maxPixelRow();
  int minPixelCol = cluster.minPixelCol();
  int minPixelRow = cluster.minPixelRow();       
  // edge method moved to topologu class
  bool edgex = (theTopol->isItEdgePixelInX(minPixelRow)) ||
    (theTopol->isItEdgePixelInX(maxPixelRow));
  bool edgey = (theTopol->isItEdgePixelInY(minPixelCol)) ||
    (theTopol->isItEdgePixelInY(maxPixelCol));

  //&&& testing...
  if (theVerboseLevel > 9) {
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
      if (thePart == GeomDetEnumerators::PixelBarrel) {
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
//       if (thePart == GeomDetEnumerators::PixelBarrel) {
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
    if (thePart == GeomDetEnumerators::PixelBarrel) { // Barrel
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
// Position estimate in X-direction
// Copy the code from CPEFromDetPosition
//-----------------------------------------------------------------------------
float 
PixelCPEInitial::xpos(const SiPixelCluster& cluster) const {

  int size = cluster.sizeX();
                                                                                
  if (size == 1) {
    float baryc = cluster.x();
    // the middle of only one pixel is equivalent to the baryc.
    // transform baryc to local
    return theTopol->localX(baryc);
  }
 
  //calculate center
  int imin = cluster.minPixelRow();
  int imax = cluster.maxPixelRow();
  float min = float(imin) + 0.5; // center of the edge
  float max = float(imax) + 0.5; // center of the edge
  float minEdge = theTopol->localX(float(imin+1)); // left inner edge
  float maxEdge = theTopol->localX(float(imax));   // right inner edge
  float center = (minEdge + maxEdge)/2.; // center of inner part
  float wInner = maxEdge-minEdge; // width of the inner part
   
  // get the charge in the edge pixels
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  float q1 = 0.;
  float q2 = 0.;
  xCharge(pixelsVec, imin, imax, q1, q2); // get q1 and q2
   
  // Estimate the charge width. main contribution + 2nd order geom corr.
  float tmp = (max+min)/2.;
  float width = (chargeWidthX() + geomCorrectionX(tmp)) * thePitchX;
  
  // Check the valid chargewidth (WHY IS THERE THE FABS??)
  float effWidth = fabs(width) - wInner;
   
  // Check the residual width
  if( (effWidth>(2*thePitchX)) || (effWidth<0.) ) { // for illiegal wifth
    effWidth=thePitchX; // make it equal to pitch
  } 

    // For X (no angles) use the MSI formula.
  // position msI
  float pos = center + (q2-q1)/(2.*(q1+q2)) * effWidth;
 
  return pos;

}
//-----------------------------------------------------------------------------
// Position estimate in the local y-direction
//-----------------------------------------------------------------------------
float 
PixelCPEInitial::ypos(const SiPixelCluster& cluster) const {

  int size = cluster.sizeY();
 
  if (size == 1) {
    float baryc = cluster.y();
    // the middle of only one pixel is equivalent to the baryc.
    // transform baryc to local
    return theTopol->localY(baryc);
  }
 
  //calculate center
  int imin = cluster.minPixelCol();
  int imax = cluster.maxPixelCol();
  //float min = float(imin) + 0.5; // center of the edge
  //float max = float(imax) + 0.5; // center of the edge
  float minEdge = theTopol->localY(float(imin+1)); // left inner edge
  float maxEdge = theTopol->localY(float(imax));   // right inner edge
  float center = (minEdge + maxEdge)/2.; // center of inner part in LC
  //float wInner = maxEdge-minEdge; // width of the inner part in LC
   
  // get the charge in the edge pixels
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  float q1 = 0.;
  float q2 = 0.;
  yCharge(pixelsVec, imin, imax, q1, q2);
    
  float pitch1 = thePitchY;
  float pitch2 = thePitchY;
  if(RectangularPixelTopology::isItBigPixelInY(imin) )
    pitch1= 2.*thePitchY;
  if(RectangularPixelTopology::isItBigPixelInY(imax) )
    pitch2= 2.*thePitchY;
   
  // position msII
  float pos = center + (q2-q1)/(2.*(q1+q2)) * (pitch1+pitch2)/2.;
  return pos;

}

//-----------------------------------------------------------------------------
// This is the main contribution to the charge width in the X direction
// Lorentz shift for the barrel and lorentz+geometry for the forward.
//-----------------------------------------------------------------------------
float PixelCPEInitial::chargeWidthX() const { 
  float chargeW = 0;
  float lorentzWidth = 2 * theLShiftX;
  if (thePart == GeomDetEnumerators::PixelBarrel) {
    chargeW = lorentzWidth; //  width from Lorentz shift
  } else { // forward
    chargeW = fabs(lorentzWidth) +                      // Lorentz shift
      theThickness * fabs(theDetR/theDetZ) / thePitchX; // + geometry
  }
  return chargeW;
}

//-----------------------------------------------------------------------------
// This is the main contribution to the charge width in the Y direction
//-----------------------------------------------------------------------------
float PixelCPEInitial::chargeWidthY() const {
  float chargeW = 0;  
  float lorentzWidth = 2 * theLShiftY;
  if (thePart == GeomDetEnumerators::PixelBarrel) {
   // Charge width comes from the geometry (inclined angles)
    chargeW = theThickness * fabs(theDetZ/theDetR) / thePitchY;
  } else { //forward
   // Width comes from geometry only, given by the tilt angle
     if ( alpha2Order) {
       chargeW = -fabs(lorentzWidth)+ theThickness * tan(20./degsPerRad) / thePitchY;
     }else {
       chargeW = theThickness * tan(20./degsPerRad) / thePitchY;
     }
  }
  return chargeW;
}

//-----------------------------------------------------------------------------
// This takes into account that the charge width is not the same across a
// single detector module (sort of a 2nd order effect).
// It makes more sense for the barrel since the barrel module are larger
// and they are placed closer top the IP.
//-----------------------------------------------------------------------------
// Correction for the X-direction
// This is better defined as the IP is well localized in the x-y plane.
float PixelCPEInitial::geomCorrectionX(float xcenter) const {
  if (thePart == GeomDetEnumerators::PixelEndcap) return 0;
  else {
    float tmp = theSign * (theThickness / theDetR) * (xcenter-theOffsetX);
    return tmp;
  }
}
// Correction for the Y-direction
// This is poorly defined becouse the IP is very smeared along the z direction.
float PixelCPEInitial::geomCorrectionY(float ycenter) const {
  if (thePart == GeomDetEnumerators::PixelEndcap) return 0;
  else {
    float tmp = (ycenter - theOffsetY) * theThickness / theDetR;
    if(theDetZ>0.) tmp = -tmp; // it is the opposite for Z>0 and Z<0
    return tmp;
  }
}
 


