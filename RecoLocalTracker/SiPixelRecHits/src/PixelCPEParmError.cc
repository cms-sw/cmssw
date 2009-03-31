// change to use Lorentz angle from DB Lotte Wilke, Jan. 31st, 2008

//#include "Utilities/Configuration/interface/Architecture.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEParmError.h"

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
PixelCPEParmError::PixelCPEParmError(edm::ParameterSet const & conf, 
				     const MagneticField *mag, const SiPixelLorentzAngle * lorentzAngle) 
  : PixelCPEBase(conf,mag,lorentzAngle)
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
//  localError() calls measurementError() after computing size and 
//  edge (flag) along x and y.
//------------------------------------------------------------------
LocalError  
PixelCPEParmError::localError( const SiPixelCluster& cluster, const GeomDetUnit & det)const 
{
  setTheDet( det, cluster );

  //--- Default is the maximum error used for edge clusters.
  float xerr = thePitchX / sqrt(12.);
  float yerr = thePitchY / sqrt(12.);

  //--- Are we near either of the edges?
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
  
  // Gavril: check if this rechit contains big pixels, 03/27/07
  bool bigInX = theTopol->containsBigPixelInX(minPixelRow, maxPixelRow);
  bool bigInY = theTopol->containsBigPixelInY(minPixelCol, maxPixelCol);

  if (edgex && edgey) {
    //--- Both axes on the edge, no point in calling PixelErrorParameterization,
    //--- just return the max errors on both.
    // return LocalError(xerr*xerr, 0,yerr*yerr);
  } else {
    pair<float,float> errPair = 
      pixelErrorParametrization_->getError(thePart, 
					   cluster.sizeX(), cluster.sizeY(), 
					   alpha_         , beta_,
					   bigInX         , bigInY); // Gavril: add big pixel flags, 03/27/07
    if (!edgex) xerr = errPair.first;
    if (!edgey) yerr = errPair.second;
  }       

  if (theVerboseLevel > 9) {
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
// Use the same as for nagles from dets case but with eta correction
//-----------------------------------------------------------------------------
float 
PixelCPEParmError::xpos(const SiPixelCluster& cluster) const {
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
  //float min = float(imin) + 0.5; // center of the edge
  //float max = float(imax) + 0.5; // center of the edge
  float minEdge = theTopol->localX(float(imin+1)); // left inner edge
  float maxEdge = theTopol->localX(float(imax));   // right inner edge
  float center = (minEdge + maxEdge)/2.; // center of inner part
  float wInner = maxEdge-minEdge; // width of the inner part
  
  // get the charge in the edge pixels
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  float q1 = 0.;
  float q2 = 0.;
  xCharge(pixelsVec, imin, imax, q1, q2); // get q1 and q2
  
  // Estimate the charge width from track angle
  float width = chargeWidthX() * thePitchX; // chargewidth still in pitch units
  
  // Check the valid chargewidth 
  float effWidth = fabs(width) - wInner;

  // Check the residual width
  if( (effWidth>(2*thePitchX)) || (effWidth<0.) ) { // for illiegal wifth
    effWidth=thePitchX; // make it equal to pitch
  }
  
  // For X (with track angles) use the MSI formula.
  // position msI
  float pos = center + (q2-q1)/(2.*(q1+q2)) * effWidth;
  

  // Delete the eta correction, it did not bring much d.k.3/07
  //if (alpha_ < 1.53) {
  //float etashift=0;
  //float charatio = q1/(q1+q2);
  //etashift = theEtaFunc.xEtaShift(size,thePitchX,charatio,alpha_);
  //pos = pos - etashift;
  //}

  return pos;
}

//-----------------------------------------------------------------------------
//  Calculates the *corrected* position of the cluster.
//  &&& Probably generic enough for the base class.
//-----------------------------------------------------------------------------
float 
PixelCPEParmError::ypos(const SiPixelCluster& cluster) const
{
 
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
  float wInner = maxEdge-minEdge; // width of the inner part in LC
    
  // get the charge in the edge pixels
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  float q1 = 0.;
  float q2 = 0.;
  yCharge(pixelsVec, imin, imax, q1, q2);
    
  // Estimate the charge width using the track angle
  float width = (chargeWidthY()) * thePitchY;
  float effWidth = fabs(width) - wInner;

  // Check the validty of the width
  if(effWidth>2*thePitchY) { //  width too large
    float edgeLength = 2*thePitchY; // take care of big pixels
    if(RectangularPixelTopology::isItBigPixelInY(imin) )
      edgeLength += thePitchY;
    if(RectangularPixelTopology::isItBigPixelInY(imax) )
      edgeLength += thePitchY;

    if(effWidth>edgeLength) effWidth=edgeLength/2.;

  } else if(effWidth<0.) { // width too small
    effWidth=thePitchY; //
  }

  // For y with track angles use msI method
  float pos = center + (q2-q1)/(2.*(q1+q2)) * effWidth;
  
  // Delete the eta correction. It did not bring much
  // eta function for shallow tracks
  //float charatio = q1/(q1+q2);
  //float etashift = theEtaFunc.yEtaShift(size,thePitchY,charatio,beta_);
  //pos = pos - etashift;

  return pos;
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
//   if (thePart == GeomDetEnumerators::PixelBarrel) {
//     // Redefine the charge width to include the offset
//     chargeW = lorentzWidth - theSign * geomCorrection() * theOffsetX;
//   } else { // forward
//     chargeW = fabs(lorentzWidth) + 
//       theThickness * fabs(theDetR/theDetZ) / thePitchX;
//   }
//   return chargeW;

  float geomWidthX = theThickness * tan(PI/2 - alpha_)/thePitchX;
  if (thePart == GeomDetEnumerators::PixelBarrel){
    return (geomWidthX) + (2 * theLShiftX);
  } else {
    return fabs(geomWidthX) + (2 * fabs(theLShiftX)); 
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
//   if (thePart == GeomDetEnumerators::PixelBarrel) {
//     chargeW = theThickness * fabs(theDetZ/theDetR) / thePitchY;
//     chargeW -= (geomCorrection() * theOffsetY);
//   } else { //forward
//     // Width comes from geometry only, fixed by the tilt angle
//    chargeW = theThickness * tan(20./degsPerRad) / thePitchY; 
//   }
//   return chargeW;

  float geomWidthY = theThickness * tan(PI/2 - beta_)/thePitchY;
  if (thePart == GeomDetEnumerators::PixelBarrel) {
    return geomWidthY;
  } else {
    return fabs(geomWidthY);
  }
}





