// OBSOLETE, DO NOT USE. 02/08 d.k.
//
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
#include "RecoLocalTracker/SiPixelRecHits/interface/CPEFromDetPosition.h"

//#define TPDEBUG
#define CORRECT_FOR_BIG_PIXELS  // Correct the MeasurementPoint & LocalPoint

// MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"

#include <iostream>
using namespace std;

const float PI = 3.141593;
const float degsPerRad = 57.29578;

//-----------------------------------------------------------------------------
//  All quantities are DetUnit-dependent, and
//  will be initialized in setTheDet().
//-----------------------------------------------------------------------------
CPEFromDetPosition::CPEFromDetPosition(edm::ParameterSet const & conf, 
				       const MagneticField *mag) {
  //--- Lorentz angle tangent per Tesla
  theTanLorentzAnglePerTesla =
    conf.getParameter<double>("TanLorentzAnglePerTesla");

  //--- Algorithm's verbosity
  theVerboseLevel = 
    conf.getUntrackedParameter<int>("VerboseLevel",0);

  //-- Magnetic Field
  magfield_ = mag;
 
  alpha2Order = conf.getParameter<bool>("Alpha2Order");

#ifdef CORRECT_FOR_BIG_PIXELS
  if (theVerboseLevel > 0) 
    LogDebug("CPEFromDetPosition")<<"Correct the Lorentz shift for big pixels";
#endif
}

//-----------------------------------------------------------------------------
//  One function to cache the variables common for one DetUnit.
//-----------------------------------------------------------------------------
void CPEFromDetPosition::setTheDet( const GeomDetUnit & det )const {
  if ( theDet == &det )
    return;       // we have already seen this det unit

  //--- This is a new det unit, so cache it
  theDet = dynamic_cast<const PixelGeomDetUnit*>( &det );
  if (! theDet) {
    // &&& Fatal error!  TO DO: throw an exception!
    assert(0);
  }

  //--- theDet->type() returns a GeomDetType, which implements subDetector()
  thePart = theDet->type().subDetector();
  switch (thePart) {
  case GeomDetEnumerators::PixelBarrel:
    // A barrel!  A barrel!
    break;
  case GeomDetEnumerators::PixelEndcap:
    // A forward!  A forward!
    break;
  default:
    LogDebug("CPEFromDetPosition") 
      << "CPEFromDetPosition:: a non-pixel detector type in here?" ;
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

  if (theVerboseLevel > 1) {
    LogDebug("CPEFromDetPosition") << "***** PIXEL LAYOUT ***** " 
				   << " thePart = " << thePart
				   << " theThickness = " << theThickness
				   << " thePitchX  = " << thePitchX 
				   << " thePitchY  = " << thePitchY 
				   << " theOffsetX = " << theOffsetX 
				   << " theOffsetY = " << theOffsetY 
				   << " theLShiftX  = " << theLShiftX;
  }
}
//----------------------------------------------------------------------
// Hit rrror in measurement coordinates
//-----------------------------------------------------------------------
MeasurementError 
CPEFromDetPosition::measurementError( const SiPixelCluster& cluster, 
				      const GeomDetUnit & det) const {
  LocalPoint lp( localPosition(cluster, det) );
  LocalError le( localError(   cluster, det) );

  return theTopol->measurementError( lp, le );
}
//-------------------------------------------------------------------------
//  Hit error in the local frame
//-------------------------------------------------------------------------
LocalError  
CPEFromDetPosition::localError( const SiPixelCluster& cluster, 
				const GeomDetUnit & det) const {
  setTheDet( det );
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
    LogDebug("CPEFromDetPosition") <<
      "Sizex = " << sizex << 
      " Sizey = " << sizey << 
      " Edgex = " << edgex << 
      " Edgey = " << edgey ;
  }

  return LocalError( err2X(edgex, sizex), 0, err2Y(edgey, sizey) );
}
//---------------------------------------------------------------------------
// Hit position in the masurement frame (in pixel/pitch units)
//--------------------------------------------------------------------------
MeasurementPoint 
CPEFromDetPosition::measurementPosition( const SiPixelCluster& cluster, 
					 const GeomDetUnit & det) const {
  if (theVerboseLevel > 9) {
    LogDebug("CPEFromDetPosition") <<
      "X-pos = " << xpos(cluster) << 
      " Y-pos = " << ypos(cluster) << 
      " Lshf = " << theLShiftX ;
  }


  // Fix to take into account the large pixels
#ifdef CORRECT_FOR_BIG_PIXELS

  cout<<"CPEFromDetPosition::measurementPosition: Not implemented "
      <<"I hope it is not needed?"<<endl;
  return MeasurementPoint(0,0);

#else

  // correct the measurement for Lorentz shift
  if ( alpha2Order) {
    float xPos = xpos(cluster); // x position in the measurement frame
    float yPos = ypos(cluster);
    float lxshift = theLShiftX; // nominal lorentz shift
    float lyshift = theLShiftY;
    if(RectangularPixelTopology::isItBigPixelInX(int(xPos))) // if big
      lxshift = theLShiftX/2.;  // reduce the shift
    if (thePart == GeomDetEnumerators::PixelBarrel) {
      lyshift =0.0;
    } else { //forward
      if(RectangularPixelTopology::isItBigPixelInY(int(yPos))) // if big
	lyshift = theLShiftY/2.;  // reduce the shift 
    }
    return MeasurementPoint( xpos(cluster)-lxshift,ypos(cluster)-lyshift);
  } else {
    float xPos = xpos(cluster); // x position in the measurement frame
    float lshift = theLShiftX; // nominal lorentz shift
    if(RectangularPixelTopology::isItBigPixelInX(int(xPos))) // if big 
      lshift = theLShiftX/2.;  // reduce the shift
    return MeasurementPoint( xpos(cluster)-lshift,ypos(cluster));
  } 
  
#endif

}
//----------------------------------------------------------------------------
// Hit position in the local frame (in cm)
//-----------------------------------------------------------------------------
LocalPoint
CPEFromDetPosition::localPosition(const SiPixelCluster& cluster, 
				  const GeomDetUnit & det) const {
  setTheDet( det );  // Initlize the det

#ifdef CORRECT_FOR_BIG_PIXELS

  float lpx = xpos(cluster);
  float lpy = ypos(cluster);
  if ( alpha2Order) {
    float lxshift = theLShiftX * thePitchX;  // shift in cm
    float lyshift = theLShiftY * thePitchY;
    if (thePart == GeomDetEnumerators::PixelBarrel) {
      LocalPoint cdfsfs(lpx-lxshift, lpy);
      return cdfsfs;
    } else { //forward
      LocalPoint cdfsfs(lpx-lxshift, lpy-lyshift);
      return cdfsfs;
    }
    
  } else {

     float lxshift = theLShiftX * thePitchX;  // shift in cm
     LocalPoint cdfsfs(lpx-lxshift, lpy );
     return cdfsfs;
  }

#else

  MeasurementPoint ssss( xpos(cluster),ypos(cluster));
  LocalPoint lp = theTopol->localPosition(ssss);
  if ( alpha2Order) {
    float lxshift = theLShiftX * thePitchX;  // shift in cm
    float lyshift = theLShiftY*thePitchY;
     if (thePart == GeomDetEnumerators::PixelBarrel) {
       LocalPoint cdfsfs(lp.x()-lxshift, lp.y());
       return cdfsfs;
     } else { //forward
       LocalPoint cdfsfs(lp.x()-lxshift, lp.y()-lyshift);
       return cdfsfs;
     }
  } else {
    float lxshift = theLShiftX * thePitchX;  // shift in cm
    LocalPoint cdfsfs(lp.x()-lxshift, lp.y() );
    return cdfsfs;
  }
  
#endif

}
//-----------------------------------------------------------------------------
// Position error estimate in X (square returned).
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::err2X(bool& edgex, int& sizex) const
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
// Position error estimate in Y (square returned).
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::err2Y(bool& edgey, int& sizey) const
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
//---------------------------------------------------------
// Main position routines
// xpos() and ypos() are split for old and new methods
#ifdef CORRECT_FOR_BIG_PIXELS

//-----------------------------------------------------------------------------
// Position estimate in X-direction, in local coordinates (cm)
//-----------------------------------------------------------------------------
float CPEFromDetPosition::xpos(const SiPixelCluster& cluster) const {
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
  vector<float> chargeVec = xCharge(pixelsVec, min, max); 
  float q1 = chargeVec[0];
  float q2 = chargeVec[1];
  
  // Estimate the charge width. main contribution + 2nd order geom corr.
  float tmp = (max+min)/2.;
  float width = (chargeWidthX() + geomCorrectionX(tmp)) * thePitchX;
  
  // Check the valid chargewidth (WHY IS THERE THE FABS??)
  float effWidth = fabs(width) - wInner;
  
  // For X (no angles) use the MSI formula.
  // position msI  
  float pos = center + (q2-q1)/(2.*(q1+q2)) * effWidth; 

  return pos;
}  // end xPos
//
float CPEFromDetPosition::ypos(const SiPixelCluster& cluster) const {
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
  float min = float(imin) + 0.5; // center of the edge
  float max = float(imax) + 0.5; // center of the edge
  float minEdge = theTopol->localY(float(imin+1)); // left inner edge 
  float maxEdge = theTopol->localY(float(imax));   // right inner edge
  float center = (minEdge + maxEdge)/2.; // center of inner part in LC
  //float wInner = maxEdge-minEdge; // width of the inner part in LC
  
  // get the charge in the edge pixels
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  vector<float> chargeVec = yCharge(pixelsVec, min, max); 
  float q1 = chargeVec[0];
  float q2 = chargeVec[1];
  
  // Estimate the charge width. main contribution + 2nd order geom corr.
  //float tmp = (max+min)/2.;
  //float width = (chargeWidthY() + geomCorrectionY(tmp)) * thePitchY;
  //float width = (chargeWidthY()) * thePitchY;  
  // Check the valid chargewidth (WHY IS THERE THE FABS??)
  //if(width<0.) cout<<" width Y < 0"<<width<<endl;
  //float effWidth = fabs(width) - wInner;

  //float pos = center + (q2*arm2-q1*arm1)/(q1+q2); // position dk  
  // position msI  
  //float pos = center + (q2-q1)/(2.*(q1+q2)) * effWidth; 

  float pitch1 = thePitchY;
  float pitch2 = thePitchY;
  if(theTopol->isItBigPixelInY(imin) ) 
    pitch1= 2.*thePitchY;
  if(theTopol->isItBigPixelInY(imax) ) 
    pitch2= 2.*thePitchY;
  
  // position msII
  float pos = center + (q2-q1)/(2.*(q1+q2)) * (pitch1+pitch2)/2.; 
  return pos;
}

#else // CORRECT_FOR_BIG_PIXELS

//-----------------------------------------------------------------------------
// Position estimate in X-direction
//-----------------------------------------------------------------------------
float CPEFromDetPosition::xpos(const SiPixelCluster& cluster) const {
  float xcluster = 0;
  int size = cluster.sizeX();
  const vector<SiPixelCluster::Pixel>& pixelsVec = cluster.pixels();
  float baryc = cluster.x();
  // &&& Testing...
  //if (baryc > 0) return baryc;

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
    // Estimate the charge width. main contribution + 2nd order geom corr.
    float chargeWX = chargeWidthX() + geomCorrectionX(xcenter);

    // Check the valid chargewidth
    float effchargeWX = fabs(chargeWX) - (float(size)-2);
    // truncated charge width only if it greather than the cluster size
    if ( (effchargeWX< 0.) || (effchargeWX>2.) ) effchargeWX = 1.;

    xcluster = xcenter + (q2-q1) * effchargeWX / (q1+q2) / 2.; // position

    // Parametrized Eta-correction. 
    // Skip it, it does not bring anything and makes forward hits worse. d.k. 6/06 
//     float alpha = estimatedAlphaForBarrel(xcenter);
//     if (alpha < 1.53) {
//       float etashift=0;
//       float charatio = q1/(q1+q2);
//       etashift = theEtaFunc.xEtaShift(size, thePitchX, 
// 				      charatio, alpha);
//       xcluster = xcluster - etashift;
//     }


  }    
  return xcluster;
}


//-----------------------------------------------------------------------------
// Position estimate in the local y-direction
//-----------------------------------------------------------------------------
float CPEFromDetPosition::ypos(const SiPixelCluster& cluster) const {
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

    //calculate charge width with/without the 2nd order geom-correction
    //float chargeWY = chargeWidthY() + geomCorrectionY(ycenter);//+correction 
    float chargeWY = chargeWidthY();  // no 2nd order correction

    float effchargeWY = fabs(chargeWY) - (float(size)-2);

    // truncate charge width 
    if ( (effchargeWY < 0.) || (effchargeWY > 1.) ) effchargeWY = 1.;

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

#endif  // CORRECT_FOR_BIG_PIXELS

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
//-----------------------------------------------------------------------------
bool CPEFromDetPosition::isFlipped() const {
  // Check the relative position of the local +/- z in global coordinates. 
  float tmp1 = theDet->surface().toGlobal(Local3DPoint(0.,0.,0.)).perp();
  float tmp2 = theDet->surface().toGlobal(Local3DPoint(0.,0.,1.)).perp();
  //cout << " 1: " << tmp1 << " 2: " << tmp2 << endl;
  if ( tmp2<tmp1 ) return true;
  else return false;    
}

//-----------------------------------------------------------------------------
// This is the main copntribution to the charge width in the X direction
// Lorentz shift for the barrel and lorentz+geometry for the forward.
//-----------------------------------------------------------------------------
float CPEFromDetPosition::chargeWidthX() const { 
  float chargeW = 0;
  float lorentzWidth = 2 * theLShiftX;
  if (thePart == GeomDetEnumerators::PixelBarrel) {  // barrel
    chargeW = lorentzWidth; // width from Lorentz shift
  } else { // forward
    chargeW = fabs(lorentzWidth) +                      // Lorentz shift 
      theThickness * fabs(theDetR/theDetZ) / thePitchX; // + geometry
  }
  return chargeW;
}

//-----------------------------------------------------------------------------
// This is the main contribution to the charge width in the Y direction
//-----------------------------------------------------------------------------
float 
CPEFromDetPosition::chargeWidthY() const
{
  float chargeW = 0;
  float lorentzWidth = 2 * theLShiftY;  
  if (thePart == GeomDetEnumerators::PixelBarrel) {  // barrel
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
float CPEFromDetPosition::geomCorrectionX(float xcenter) const { 
  if (thePart == GeomDetEnumerators::PixelEndcap) return 0;
  else {
    float tmp = theSign * (theThickness / theDetR) * (xcenter-theOffsetX);
    return tmp;
  }
}
// Correction for the Y-direction
// This is poorly defined becouse the IP is very smeared along the z direction.
float CPEFromDetPosition::geomCorrectionY(float ycenter) const { 
  if (thePart == GeomDetEnumerators::PixelEndcap) return 0;
  else { 
    float tmp = (ycenter - theOffsetY) * theThickness / theDetR;
    if(theDetZ>0.) tmp = -tmp; // it is the opposite for Z>0 and Z<0
    return tmp;
  }
}

//-----------------------------------------------------------------------------
// Lorentz shift. For the moment only in X direction (barrel & endcaps)
// For the forward the y componenet might have to be added.
//-----------------------------------------------------------------------------
float CPEFromDetPosition::lorentzShiftX() const {

  LocalVector dir = driftDirection(magfield_->inTesla(theDet->surface().position()) );

  // max shift in cm 
  float xdrift = dir.x()/dir.z() * theThickness;  
  // express the shift in units of pitch, 
  // divide by 2 to get the hit correction
  float lshift = xdrift / thePitchX / 2.; 

  //cout << "Lorentz Drift = " << lshift << endl;
  //cout << "X Drift = " << dir.x() << endl;
  //cout << "Z Drift = " << dir.z() << endl;
 
  return lshift;  
}

float CPEFromDetPosition::lorentzShiftY() const {

  LocalVector dir = driftDirection(magfield_->inTesla(theDet->surface().position()) );
  float ydrift = dir.y()/dir.z() * theThickness;
  float lshift = ydrift / thePitchY / 2.;
  return lshift;
}


//-----------------------------------------------------------------------------
// unused
//-----------------------------------------------------------------------------
// float 
// CPEFromDetPosition::chaWidth2X(const float& centerx) const
// {
//   float chargeWX = chargeWidthX() + theSign * geomCorrection() * centerx;
//   if ( chargeWX > 1. || chargeWX <= 0. ) chargeWX=1.;
//   return chargeWX;
// }

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
//  Works OK for barrel and forward.
//  The formulas used for dir_x,y,z have to be exactly the same as the ones 
//  used in the digitizer (SiPixelDigitizerAlgorithm.cc).
//  Assumption: setTheDet() has been called already.
//-----------------------------------------------------------------------------
LocalVector 
CPEFromDetPosition::driftDirection( GlobalVector bfield ) const {
  Frame detFrame(theDet->surface().position(), theDet->surface().rotation());
  LocalVector Bfield = detFrame.toLocal(bfield);

  float alpha2;
  if ( alpha2Order) {
     alpha2 = theTanLorentzAnglePerTesla*theTanLorentzAnglePerTesla;
  }else {
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

  if (theVerboseLevel > 9) { 
    LogDebug("CPEFromDetPosition") 
      << " The drift direction in local coordinate is " 
      << theDriftDirection    ;
  }

  return theDriftDirection;
}




