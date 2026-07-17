#ifndef DataFormats_FTLRecHit_FTLMergedCluster_h
#define DataFormats_FTLRecHit_FTLMergedCluster_h

#include <vector>
#include <unordered_set>
#include <algorithm>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetRefVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

class FTLMergedCluster {
public:
  // Default constructor
  constexpr FTLMergedCluster()
      : id_(0), energy_(0.0), time_(0.0), timeError_(0.0), x_(0.0), y_(0.0), xError_(0.0), yError_(0.0) {}

  // Constructor
  FTLMergedCluster(DetId id,
                   float energy,
                   float time,
                   float timeError,
                   float x,
                   float y,
                   float xError,
                   float yError,
                   const std::vector<DetId>& clusterIds,
                   const std::vector<FTLClusterRef>& clusterRefs)
      : id_(id),
        energy_(energy),
        time_(time),
        timeError_(timeError),
        x_(x),
        y_(y),
        xError_(xError),
        yError_(yError),
        clusterIds_(clusterIds),
        clusterRefs_(clusterRefs) {}

  FTLMergedCluster(DetId id, float energy, float time, float timeError, float x, float y, float xError, float yError)
      : id_(id), energy_(energy), time_(time), timeError_(timeError), x_(x), y_(y), xError_(xError), yError_(yError) {}

  // Functions to access the data members
  DetId id() const { return id_; }
  float energy() const { return energy_; }
  float time() const { return time_; }
  float timeError() const { return timeError_; }
  float x() const { return x_; }
  float y() const { return y_; }
  float xError() const { return xError_; }
  float yError() const { return yError_; }
  const std::vector<DetId>& clusterIds() const { return clusterIds_; }
  const std::vector<FTLClusterRef>& clusterRefs() const { return clusterRefs_; }
  size_t nClusters() const {
    if (hitsDetId_.size() == 1) {
      return 1;
    }
    std::unordered_set<DetId> seen;
    for (const auto& x : hitsDetId_) {
      seen.insert(x);
    }
    return seen.size();
  }

  void addHit(
      const DetId& id, const int row, const int col, const float time, const float timeError, const float energy) {
    hitsDetId_.push_back(id);
    uint8_t row8 = static_cast<uint8_t>(std::clamp(row, 0, 15));
    uint8_t col8 = static_cast<uint8_t>(std::clamp(col, 0, 15));
    uint8_t rowcol = static_cast<uint8_t>((row8 << krcOffset) | col8);
    hitsRowCol_.push_back(rowcol);
    hitsTime_.push_back(time);
    hitsTimeError_.push_back(timeError);
    hitsEnergy_.push_back(energy);
  }

  size_t size() const { return hitsDetId_.size(); }
  size_t nHits() const { return hitsDetId_.size(); }
  DetId hDetId(size_t index) const { return hitsDetId_[index]; }
  uint32_t hRow(size_t index) const { return static_cast<uint32_t>((hitsRowCol_[index] >> krcOffset) & krcMask); }
  uint32_t hCol(size_t index) const { return static_cast<uint32_t>(hitsRowCol_[index] & krcMask); }
  // encode in a single uint64_t both DetId rawId and RowCol for hit comparison
  uint64_t hUniqueId(size_t index) const {
    return static_cast<uint64_t>(hitsDetId_[index].rawId()) << 8 | static_cast<uint64_t>(hitsRowCol_[index]);
  }
  float hTime(size_t index) const { return hitsTime_[index]; }
  float hTimeError(size_t index) const { return hitsTimeError_[index]; }
  float hEnergy(size_t index) const { return hitsEnergy_[index]; }

  bool operator==(FTLMergedCluster const& lh) const {
    bool isSame = nHits() == lh.nHits();
    if (isSame) {
      for (size_t index = 0; index < nHits(); index++) {
        isSame = isSame && (hUniqueId(index) == lh.hUniqueId(index));
      }
    }
    return isSame;
  }

private:
  static constexpr uint32_t krcOffset = 4;
  static constexpr uint32_t krcMask = 0x0F;

  DetId id_;
  float energy_;
  float time_;
  float timeError_;
  float x_;
  float y_;
  float xError_;
  float yError_;

  std::vector<DetId> clusterIds_;
  std::vector<FTLClusterRef> clusterRefs_;

  std::vector<DetId> hitsDetId_;
  std::vector<uint8_t> hitsRowCol_;
  std::vector<float> hitsTime_;
  std::vector<float> hitsTimeError_;
  std::vector<float> hitsEnergy_;
};

std::ostream& operator<<(std::ostream& s, const FTLMergedCluster& clu);

#endif
