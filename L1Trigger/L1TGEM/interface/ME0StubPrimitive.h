#ifndef L1Trigger_L1TGEM_ME0StubPrimitive_H
#define L1Trigger_L1TGEM_ME0StubPrimitive_H

#include <vector>
#include <cstdint>
#include <string>
#include <iostream>

#include "L1Trigger/L1TGEM/interface/ME0StubFit.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

class ME0StubPrimitive final {
public:
  // Constructors
  ME0StubPrimitive();
  ME0StubPrimitive(int layerCount, int hitCount, int patternId, int strip, int etaPartition);
  ME0StubPrimitive(int layerCount, int hitCount, int patternId, int strip, int etaPartition, double bx);
  ME0StubPrimitive(
      int layerCount, int hitCount, int patternId, int strip, int etaPartition, double bx, std::vector<int>& centroids);

  // clone
  ME0StubPrimitive* clone() const { return new ME0StubPrimitive(*this); }

  // Get private variable
  int layerCount() const { return layerCount_; }
  int hitCount() const { return hitCount_; }
  int patternId() const { return patternId_; }
  int strip() const { return strip_; }
  int etaPartition() const { return etaPartition_; }
  int bx() const { return bx_; }
  double subStrip() const { return subStrip_; }
  double bendingAngle() const { return bendingAngle_; }
  double mse() const { return mse_; }
  std::vector<int> centroids() const { return centroids_; }
  int quality() const { return quality_; }
  int maxClusterSize() const { return maxClusterSize_; }
  int maxNoise() const { return maxNoise_; }

  // Set private variable
  void setLayerCount(int layerCount) { layerCount_ = layerCount; }
  void setHitCount(int hitCount) { hitCount_ = hitCount; }
  void setPatternId(int patternId) { patternId_ = patternId; }
  void setStrip(int strip) { strip_ = strip; }
  void setEtaPartition(int etaPartition) { etaPartition_ = etaPartition; }
  void setBx(double bx) { bx_ = bx; }
  void setCentroids(std::vector<int> centroids) { centroids_ = centroids; }
  void setMaxClusterSize(int maxClusterSize) { maxClusterSize_ = maxClusterSize; }
  void setMaxNoise(int maxNoise) { maxNoise_ = maxNoise; }

  void reset();
  void updateQuality();
  void fit(int patSpan);

  // operators
  bool operator==(const ME0StubPrimitive& other) {
    if (layerCount_ == 0 && other.layerCount_ == 0) {
      return true;
    }
    return (quality_ == other.quality_);
  }
  bool operator>(const ME0StubPrimitive& other) { return (quality_ > other.quality_); }
  bool operator<(const ME0StubPrimitive& other) { return (quality_ < other.quality_); }
  bool operator>=(const ME0StubPrimitive& other) { return (quality_ >= other.quality_); }
  bool operator<=(const ME0StubPrimitive& other) { return (quality_ <= other.quality_); }
  // ostream
  friend std::ostream& operator<<(std::ostream& os, const ME0StubPrimitive& stub) {
    os << "id=" << stub.patternId() << ", lc=" << stub.layerCount() << ", strip=" << stub.strip()
       << ", prt=" << stub.etaPartition() << ", quality=" << stub.quality();
    return os;
  }

private:
  int layerCount_, hitCount_, patternId_, strip_, etaPartition_;
  double bx_ = -9999;
  std::vector<int> centroids_;
  double subStrip_ = 0.0;
  double bendingAngle_ = 0.0;
  double mse_ = 9999;
  int quality_ = 0;
  int maxClusterSize_ = 0;
  int maxNoise_ = 0;
  bool ignoreBend_ = false;
};

#endif
