#ifndef RecoTracker_LST_interface_LSTOutput_h
#define RecoTracker_LST_interface_LSTOutput_h

#include <memory>
#include <vector>

#include "RecoTracker/LSTCore/interface/Common.h"

class LSTOutput {
public:
  LSTOutput() = default;
  LSTOutput(std::vector<std::vector<unsigned int>> const hitIdx,
            std::vector<unsigned int> const len,
            std::vector<int> const seedIdx,
            std::vector<short> const trackCandidateType)
      : hitIdx_(std::move(hitIdx)),
        len_(std::move(len)),
        seedIdx_(std::move(seedIdx)),
        trackCandidateType_(std::move(trackCandidateType)) {}

  using LSTTCType = lst::LSTObjType;

  // Hit indices of each of the LST track candidates.
  std::vector<std::vector<unsigned int>> const& hitIdx() const { return hitIdx_; }
  // Number of hits of each of the LST track candidates.
  std::vector<unsigned int> const& len() const { return len_; }
  // Index of the pixel track associated to each of the LST track candidates.
  // If not associated to a pixel track, which is the case for T5s, it defaults to -1.
  std::vector<int> const& seedIdx() const { return seedIdx_; }
  // LSTTCType from RecoTracker/LSTCore/interface/Common.h
  std::vector<short> const& trackCandidateType() const { return trackCandidateType_; }

private:
  std::vector<std::vector<unsigned int>> hitIdx_;
  std::vector<unsigned int> len_;
  std::vector<int> seedIdx_;
  std::vector<short> trackCandidateType_;
};

#endif
