#ifndef DTTrigGeomUtils_H
#define DTTrigGeomUtils_H

/*
 * \file DTTrigGeomUtils.h
 *
 * $Date: 2009/08/03 16:08:37 $
 * $Revision: 1.2 $
 * \author C. Battilana - CIEMAT
 *
*/

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/Pi.h"

#include<cmath>

class DTGeometry;
class DTRecSegment4D;
class DTChamberId;
class L1MuDTChambPhDigi;

class DTTrigGeomUtils {
  
 public:
 
  /// Constructor
  DTTrigGeomUtils(edm::ESHandle<DTGeometry> muonGeom, bool dirInDeg=true);
  
  /// Destructor
  virtual ~DTTrigGeomUtils();
  
  /// Compute phi range in local chamber coordinates
  void phiRange(const DTChamberId& id, float& min, float& max, int&nbins, float step=15);

  /// Compute theta range in local chamber coordinates
  void thetaRange(const DTChamberId& id, float& min, float& max, int& nbins, float step=15);

  /// Compute track coordinates with SC sector numbering
  void computeSCCoordinates(const DTRecSegment4D* track, int& scsec, float& x, float& xdir, float& y, float& ydir);

  /// Return local position (trigger RF) for a given trigger primitive
  float trigPos(const L1MuDTChambPhDigi* trig);

  /// Return local direction (trigger RF) for a given trigger primitive
  float trigDir(const L1MuDTChambPhDigi* trig);

  /// Compute Trigger x coordinate in chamber RF
  void trigToSeg(int st, float& x, float dir) { x -= tan(dir/radToDeg_)*zcn_[st-1]; };

  /// Checks id the chamber has positive RF;
  bool hasPosRF(int wh, int sec) { return wh>0 || (wh==0 && sec%4>1); };

 private:
  
  edm::ESHandle<DTGeometry> muonGeom_;
  float  zcn_[4];
  float radToDeg_;
  float xCenter_[2]; // 0=4/13 - 1=10/14

};

#endif
