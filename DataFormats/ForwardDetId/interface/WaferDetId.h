#ifndef DataFormats_ForwardDetId_WaferDetId_H
#define DataFormats_ForwardDetId_WaferDetId_H 1

#include "DataFormats/DetId/interface/DetId.h"

class WaferDetId : public DetId {

public:

  WaferDetId();
  WaferDetId(uint32_t id);
  WaferDetId(DetId::Detector det, int subdet);

  virtual ~WaferDetId() { }

  /// get the type
  virtual int type() const { return 0; }

  /// get the z-side of the cell (1/-1)
  virtual int zside() const { return 1; }

  /// get the layer #
  virtual int layer() const { return 0; }

  /// get the cell #'s in u,v or in x,y
  virtual int cellU() const { return 0; }
  virtual int cellV() const { return 0; }
  virtual int getN()  const { return 0; }
  std::pair<int,int> cellUV() const { return std::pair<int,int>(cellU(),cellV()); }
  int cellX() const { return (3*(cellV()-getN())+2); }
  int cellY() const { return (2*cellU()-(getN()+cellV())); }
  std::pair<int,int> cellXY() const { return std::pair<int,int>(cellX(),cellY()); }

  /// get the wafer #'s in u,v or in x,y
  virtual int waferUAbs() const { return 0; }
  virtual int waferVAbs() const { return 0; }
  virtual int waferU()    const { return 0; }
  virtual int waferV()    const { return 0; }
  std::pair<int,int> waferUV() const { return std::pair<int,int>(waferU(),waferV()); }
  int waferX() const { return (-2*waferU()+waferV()); }
  int waferY() const { return (2*waferV()); }
  std::pair<int,int> waferXY() const { return std::pair<int,int>(waferX(),waferY()); }

  /// consistency check : no bits left => no overhead
  virtual bool isEE() const  { return true; }
  virtual bool isHE() const  { return false; }
};

#endif
