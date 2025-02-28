#ifndef DataFormats_GEMDigi_ME0Stub_H
#define DataFormats_GEMDigi_ME0Stub_H

#include <vector>
#include <cstdint>
#include <string>
#include <iostream>
#include <iomanip>

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "L1Trigger/L1TGEM/interface/ME0StubPrimitive.h"

class ME0Stub final {
public:
  ME0Stub()
      : detId_(), etaPartition_(0), padStrip_(0), bendingAngle_(0), layerCount_(0), quality_(0), patternId_(0), bx_(0) {}
  ME0Stub(const GEMDetId& id, const ME0StubPrimitive& stub)
      : detId_(id),
        etaPartition_(stub.etaPartition()),
        padStrip_(stub.strip() + stub.subStrip()),
        bendingAngle_(stub.bendingAngle()),
        layerCount_(stub.layerCount()),
        quality_(stub.quality()),
        patternId_(stub.patternId()),
        bx_(stub.bx()) {}
  ME0Stub(const GEMDetId& id,
          int etaPartition,
          double padStrip,
          double bendingAngle,
          int layerCount,
          int quality,
          int patternId,
          double bx)
      : detId_(id),
        etaPartition_(etaPartition),
        padStrip_(padStrip),
        bendingAngle_(bendingAngle),
        layerCount_(layerCount),
        quality_(quality),
        patternId_(patternId),
        bx_(bx) {}

  // clone
  ME0Stub* clone() const { return new ME0Stub(*this); }

  // Get private variable
  GEMDetId detId() const { return detId_; }
  int etaPartition() const { return etaPartition_; }
  double strip() const { return padStrip_; }
  double bendingAngle() const { return bendingAngle_; }
  int layerCount() const { return layerCount_; }
  int quality() const { return quality_; }
  int patternId() const { return patternId_; }
  double bx() const { return bx_; }

  // operators
  bool operator==(const ME0Stub& other) {
    if (layerCount_ == 0 && other.layerCount_ == 0) {
      return true;
    }
    return (quality_ == other.quality_);
  }
  bool operator>(const ME0Stub& other) { return (quality_ > other.quality_); }
  bool operator<(const ME0Stub& other) { return (quality_ < other.quality_); }
  bool operator>=(const ME0Stub& other) { return (quality_ >= other.quality_); }
  bool operator<=(const ME0Stub& other) { return (quality_ <= other.quality_); }
  // ostream
  friend std::ostream& operator<<(std::ostream& os, const ME0Stub& stub) {
    os << "id=" << stub.patternId() << ", lc=" << stub.layerCount() << ", strip=" << std::fixed << std::setprecision(3)
       << stub.strip() << ", prt=" << stub.etaPartition() << ", quality=" << stub.quality();
    return os;
  }

private:
  GEMDetId detId_;
  int etaPartition_;
  double padStrip_;
  double bendingAngle_;
  int layerCount_;
  int quality_;
  int patternId_;
  double bx_;
};

#endif