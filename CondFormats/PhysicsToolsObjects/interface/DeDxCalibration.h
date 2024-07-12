#ifndef CondFormats_PhysicsToolsObjects_DeDxCalibration_h
#define CondFormats_PhysicsToolsObjects_DeDxCalibration_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
class DeDxCalibration {
public:
  DeDxCalibration();
  virtual ~DeDxCalibration() {}

  typedef std::pair<uint32_t, unsigned char> ChipId;
  DeDxCalibration(const std::vector<double>& thr,
                  const std::vector<double>& alpha,
                  const std::vector<double>& sigma,
                  const std::map<ChipId, float>& gain)
      : thr_(thr), alpha_(alpha), sigma_(sigma), gain_(gain){};

  const std::vector<double>& thr() const { return thr_; }
  const std::vector<double>& alpha() const { return alpha_; }
  const std::vector<double>& sigma() const { return sigma_; }
  const std::map<ChipId, float>& gain() const { return gain_; }

  void setThr(const std::vector<double>& v) { thr_ = v; }
  void setAlpha(const std::vector<double>& v) { alpha_ = v; }
  void setSigma(const std::vector<double>& v) { sigma_ = v; }
  void setGain(const std::map<ChipId, float>& v) { gain_ = v; }

private:
  std::vector<double> thr_;
  std::vector<double> alpha_;
  std::vector<double> sigma_;
  std::map<ChipId, float> gain_;

  COND_SERIALIZABLE;
};
#endif
