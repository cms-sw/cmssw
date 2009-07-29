#ifndef DTTrigGeomUtils_H
#define DTTrigGeomUtils_H

/*
 * \file DTTrigGeomUtils.h
 *
 * $Date: 2009/04/09 15:44:50 $
 * $Revision: 1.1 $
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
  
  /// Calculate phi range in local chamber coordinates
  void phiRange(const DTChamberId& id, float& min, float& max);

  /// Calculate theta range in local chamber coordinates
  void thetaRange(const DTChamberId& id, float& min, float& max);

  /// Compute track coordinates with SC sector numbering
  void computeSCCoordinates(const DTRecSegment4D* track, int& scsec, float& x, float& xdir, float& y, float& ydir);

  /// Return local position (chamber RF) for a given trigger segment
  float trigPos(const L1MuDTChambPhDigi* trig);

  /// Return local direction (chamber RF) for a given trigger segment
  float trigDir(const L1MuDTChambPhDigi* trig);

  /// Propagate Trigger x coordinate to chamber z coordinate
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
