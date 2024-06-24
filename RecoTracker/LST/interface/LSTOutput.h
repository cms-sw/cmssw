#ifndef RecoTracker_LST_LSTOutput_h
#define RecoTracker_LST_LSTOutput_h

#include <memory>
#include <vector>

class LSTOutput {
public:
  LSTOutput() = default;
  LSTOutput(std::vector<std::vector<unsigned int>> hitIdx,
            std::vector<unsigned int> len,
            std::vector<int> seedIdx,
            std::vector<short> trackCandidateType) {
    hitIdx_ = std::move(hitIdx);
    len_ = std::move(len);
    seedIdx_ = std::move(seedIdx);
    trackCandidateType_ = std::move(trackCandidateType);
  }

  ~LSTOutput() = default;

  enum LSTTCType { T5 = 4, pT3 = 5, pT5 = 7, pLS = 8 };

  std::vector<std::vector<unsigned int>> const& hitIdx() const { return hitIdx_; }
  std::vector<unsigned int> const& len() const { return len_; }
  std::vector<int> const& seedIdx() const { return seedIdx_; }
  std::vector<short> const& trackCandidateType() const { return trackCandidateType_; }

private:
  std::vector<std::vector<unsigned int>> hitIdx_;
  std::vector<unsigned int> len_;
  std::vector<int> seedIdx_;
  std::vector<short> trackCandidateType_;
};

#endif
