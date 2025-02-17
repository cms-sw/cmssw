#ifndef DataFormats_ME0Stub_H
#define DataFormats_ME0Stub_H

#include <vector>
#include <cstdint>
#include <string>
#include <iostream>
#include <iomanip>

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"

class ME0Stub final {
public:
  ME0Stub() : detId(), etaPartition(0), padStrip(0), bending_angle(0), layercount(0), quality(0), patternId(0), bx(0) {}
  ME0Stub(const GEMDetId& id, const ME0StubPrimitive& stub)
      : detId(id),
        etaPartition(stub.EtaPartition()),
        padStrip(stub.Strip() + stub.SubStrip()),
        bending_angle(stub.BendingAngle()),
        layercount(stub.LayerCount()),
        quality(stub.Quality()),
        patternId(stub.PatternId()),
        bx(stub.BX()) {}
  ME0Stub(const GEMDetId& id,
          int etaPartition_,
          double padStrip_,
          double bending_angle_,
          int layercount_,
          int quality_,
          int patternId_,
          double bx_)
      : detId(id),
        etaPartition(etaPartition_),
        padStrip(padStrip_),
        bending_angle(bending_angle_),
        layercount(layercount_),
        quality(quality_),
        patternId(patternId_),
        bx(bx_) {}

  // clone
  ME0Stub* clone() const { return new ME0Stub(*this); }

  // Get private variable
  GEMDetId DetId() const { return detId; }
  int EtaPartition() const { return etaPartition; }
  double Strip() const { return padStrip; }
  double BendingAngle() const { return bending_angle; }
  int LayerCount() const { return layercount; }
  int Quality() const { return quality; }
  int PatternId() const { return patternId; }
  double BX() const { return bx; }

  // operators
  bool operator==(const ME0Stub& other) {
    if (layercount == 0 && other.layercount == 0) {
      return true;
    }
    return (quality == other.quality);
  }
  bool operator>(const ME0Stub& other) { return (quality > other.quality); }
  bool operator<(const ME0Stub& other) { return (quality < other.quality); }
  bool operator>=(const ME0Stub& other) { return (quality >= other.quality); }
  bool operator<=(const ME0Stub& other) { return (quality <= other.quality); }
  // ostream
  friend std::ostream& operator<<(std::ostream& os, const ME0Stub& stub) {
    os << "id=" << stub.PatternId() << ", lc=" << stub.LayerCount() << ", strip=" << std::fixed << std::setprecision(3)
       << stub.Strip() << ", prt=" << stub.EtaPartition() << ", quality=" << stub.Quality();
    return os;
  }

private:
  GEMDetId detId;
  int etaPartition;
  double padStrip;
  double bending_angle;
  int layercount;
  int quality;
  int patternId;
  double bx;
};

#endif