// Include our own header first
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPETemplateReco.h"

// Geometry services
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

//#define DEBUG

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

#include <iostream>
using namespace std;

const float PI = 3.141593;
const float HALFPI = PI * 0.5;
const float degsPerRad = 57.29578;


// The template header files
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#include <vector>
#include "boost/multi_array.hpp"
using namespace SiPixelTemplateReco ;


//-----------------------------------------------------------------------------
//  Constructor.  All detUnit-dependent quantities will be initialized later,
//  in setTheDet().  Here we only load the templates into the template store templ_ .
//-----------------------------------------------------------------------------
PixelCPETemplateReco::PixelCPETemplateReco(edm::ParameterSet const & conf, 
				     const MagneticField *mag) 
  : PixelCPEBase(conf,mag)
{
  // Initialize template store, CMSSW simulation as thePixelTemp[0]
  templ_.pushfile(201);

  // Initialize template store, Pixelav 125V simulation as
  // thePixelTemp[1]
  templ_.pushfile(1);
	 
  // Initialize template store, CMSSW simulation w/ reduced difusion
  // as thePixelTemp[2]
  templ_.pushfile(401);

}

//-----------------------------------------------------------------------------
//  Clean up.
//-----------------------------------------------------------------------------
PixelCPETemplateReco::~PixelCPETemplateReco()
{
  // &&& delete template store?
}


//------------------------------------------------------------------
//  Public methods mandated by the base class.
//------------------------------------------------------------------


//------------------------------------------------------------------
//  localPosition() calls measurementPosition() and then converts it
//  to the LocalPoint. USE THE ONE FROM THE BASE CLASS
//------------------------------------------------------------------
// LocalPoint
// PixelCPETemplateReco::localPosition(const SiPixelCluster& cluster, 
// 				 const GeomDetUnit & det) const {
//   setTheDet( det );
//   MeasurementPoint ssss = measurementPosition(cluster, det);
//   LocalPoint cdfsfs = theTopol->localPosition(ssss);
//   return cdfsfs;
// }
//------------------------------------------------------------------
//  localError() calls measurementError() after computing size and 
//  edge (flag) along x and y.
//------------------------------------------------------------------
LocalError  
PixelCPETemplateReco::localError( const SiPixelCluster& cluster, const GeomDetUnit & det)const 
{
  setTheDet( det );

  //--- Default is the maximum error used for edge clusters.
  float xerr = thePitchX / sqrt(12.);
  float yerr = thePitchY / sqrt(12.);

  //--- Are we near either of the edges?
  //bool edgex = (cluster.edgeHitX()) || (cluster.maxPixelRow() > theNumOfRow);//wrong
  //bool edgey = (cluster.edgeHitY()) || (cluster.maxPixelCol() > theNumOfCol); 

  bool edgex = (cluster.minPixelRow()==0) ||  // use min and max pixels
    (cluster.maxPixelRow()==(theNumOfRow-1));
  bool edgey = (cluster.minPixelCol()==0) ||
    (cluster.maxPixelCol()==(theNumOfCol-1));

  if (edgex && edgey) {
    //--- Both axes on the edge, no point in calling PixelErrorParameterization,
    //--- just return the max errors on both.
    cout << "PixelCPETemplateReco::localError: edge hit, returning sqrt(12)." 
	 << endl;
    return LocalError(xerr*xerr, 0,yerr*yerr);
  }
  else {
    // &&& need a class const
    static const float micronsToCm = 1e-4;

    xerr = templSigmaX_ * micronsToCm;
    yerr = templSigmaY_ * micronsToCm;

    // &&& should also check ierr (saved as class variable) and return
    // &&& nonsense (another class static) if the template fit failed.
  }       

  if (theVerboseLevel > 9) {
    LogDebug("PixelCPETemplateReco") <<
      "Sizex = " << cluster.sizeX() << " Sizey = " << cluster.sizeY() << 
      " Edgex = " << edgex          << " Edgey = " << edgey << 
      " ErrX = " << xerr            << " ErrY  = " << yerr;
  }
  return LocalError(xerr*xerr, 0,yerr*yerr);
}



//-----------------------------------------------------------------------------
//!  The method in which we perform the template hit reconstruction, via a call 
//!  to PixelTemplateReco2D() function.  In setting up the call, we transfer
//!  the digis from SiPixelCluster cluster to a 2D matrix (in fact a
//!  boost::multi_array<float,2>) called clust_array_2d.  En route we also flag the
//!  double pixels we find while filling the 2D matrix, and store that in 
//!  xdouble and ydouble vectors.
//!
//!  Questions for Morris: given that the size of the template matrix is 
//!  constrained to 21x7 pixels, why not use arrays instead of vectors for xdouble
//!  and ydouble (since there's nothing dynamic about them, they need to be 
//!  filled whole...) and why not use a 2D array instead of boost::multi_array?
//-----------------------------------------------------------------------------
MeasurementPoint 
PixelCPETemplateReco::measurementPosition( const SiPixelCluster& cluster, 
					   const GeomDetUnit & det) const 
{
  if (theVerboseLevel > 9) {
    LogDebug("PixelCPETemplateReco") <<
      "::measurementPosition: processing cluster at" << 
      "X-pos = " << xpos(cluster) << 
      " Y-pos = " << ypos(cluster);
  }

  int ierr;   //!< return status
  int ID = 2; //!< picks the third entry from the template store, namely 401
  bool fpix;  //!< barrel(false) or forward(true)
  if (thePart == GeomDetEnumerators::PixelBarrel)   
    fpix = false;    // no, it's not forward -- it's barrel
  else                                              
    fpix = true;     // yes, it's forward

  // Make cot(alpha) and cot(beta)... cot(x) = tan(pi/2 - x);
  float cotalpha = tan(HALFPI - alpha_);  
  float cotbeta  = tan(HALFPI - beta_);   

  // Make from cluster (a SiPixelCluster) a boost multi_array_2d called 
  // clust_array_2d.
  boost::multi_array<float, 2> clust_array_2d(boost::extents[7][21]);

  // Copy clust's pixels (calibrated in electrons) into clust_array_2d;


  // Preparing to retrieve ADC counts from the SiPixelCluster.  In the cluster,
  // we have the following:
  //   int minPixelRow(); // Minimum pixel index in the x direction (low edge).
  //   int maxPixelRow(); // Maximum pixel index in the x direction (top edge).
  //   int minPixelCol(); // Minimum pixel index in the y direction (left edge).
  //   int maxPixelCol(); // Maximum pixel index in the y direction (right edge).
  // So the pixels from minPixelRow() will go into clust_array_2d[0][*],
  // and the pixels from minPixelCol() will go into clust_array_2d[*][0].
  int row_offset = cluster.minPixelRow();
  int col_offset = cluster.minPixelCol();

  const std::vector<SiPixelCluster::Pixel> & pixVec = cluster.pixels();
  std::vector<SiPixelCluster::Pixel>::const_iterator 
    pixIter = pixVec.begin(), pixEnd = pixVec.end();
  //
  for ( ; pixIter != pixEnd; ++pixIter ) {
    //
    // *pixIter dereferences to Pixel struct, with public vars x, y, adc (all float)
    int irow = int(pixIter->x) - row_offset;   // &&& do we need +0.5 ???
    int icol = int(pixIter->y) - col_offset;   // &&& do we need +0.5 ???
    clust_array_2d[ irow ][ icol ] = pixIter->adc;
  }

  // Make and fill the bool arrays flagging double pixels
  // &&& Need to define constants for 7 and 21 somewhere!
  std::vector<bool> ydouble(21), xdouble(7);
  // x directions (shorter), rows
  for (int irow = 0; irow < 7; ++irow) {
    xdouble[irow] = RectangularPixelTopology::isItBigPixelInX( irow+row_offset );
  }
  // y directions (longer), columns
  for (int icol = 0; icol < 21; ++icol) {
    ydouble[icol] = RectangularPixelTopology::isItBigPixelInY( icol+col_offset );
  }

  // Output:
  static float nonsense = -99999.0; // nonsense init value
  templXrec_ = templYrec_ = templSigmaX_ = templSigmaY_ = nonsense;

  // ******************************************************************
  // Do it!
  ierr = 
    PixelTempReco2D(ID, fpix, cotalpha, cotbeta, 
		    clust_array_2d, ydouble, xdouble, 
		    templ_, 
		    templYrec_, templSigmaY_,
		    templXrec_, templSigmaX_);
  // ******************************************************************


  // Check exit status
  if (ierr != 0) {
    printf("reconstruction failed with error %d \n", ierr);
    // &&&  throw an exception?
    return MeasurementPoint( nonsense, nonsense );
  }

  // &&& need a class const
  static const float micronsToCm = 1e-4;

  // The template code returns the answer in microns.  We need to return in
  // the units of pixel size, so we convert first to cm, then divide by the pitch:
  float xpos_in_pixels = templXrec_ * micronsToCm / thePitchX + row_offset + 0.5;
  float ypos_in_pixels = templYrec_ * micronsToCm / thePitchY + col_offset + 0.5;


  return MeasurementPoint( xpos_in_pixels, ypos_in_pixels );

}


//------------------------------------------------------------------
//  Helper methods (protected)
//------------------------------------------------------------------

//-----------------------------------------------------------------------------
//  Calculates the *corrected* position of the cluster.
//  &&& Probably generic enough for the base class.
//-----------------------------------------------------------------------------
float 
PixelCPETemplateReco::xpos(const SiPixelCluster& cluster) const
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
PixelCPETemplateReco::ypos(const SiPixelCluster& cluster) const
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
PixelCPETemplateReco::chargeWidthX() const
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
PixelCPETemplateReco::chargeWidthY() const
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





