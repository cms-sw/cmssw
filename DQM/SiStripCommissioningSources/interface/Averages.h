#ifndef DQM_SiStripCommissioningSources_Averages_H
#define DQM_SiStripCommissioningSources_Averages_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <vector>
#include <map>
#include <cstdint>

/** */
class Averages {
public:
  Averages();
  ~Averages() { ; }

  class Params {
  public:
    float mean_;
    float median_;
    float mode_;
    float rms_;
    float weight_;
    float max_;
    float min_;
    uint32_t num_;
    Params()
        : mean_(1. * sistrip::invalid_),
          median_(1. * sistrip::invalid_),
          mode_(1. * sistrip::invalid_),
          rms_(1. * sistrip::invalid_),
          weight_(1. * sistrip::invalid_),
          max_(-1. * sistrip::invalid_),
          min_(1. * sistrip::invalid_),
          num_(sistrip::invalid_) {
      ;
    }
    ~Params() { ; }
  };

  /** */
  void add(const float& value, const float& weight);
  void add(const float& value);
  /** */
  void add(const uint32_t& value, const uint32_t& weight);
  void add(const uint32_t& value);

  /** */
  void calc(Params&);

private:
  uint32_t n_;
  float s_;
  float x_;
  float xx_;
  std::vector<float> median_;
  std::map<uint32_t, uint32_t> mode_;
};

#endif  // DQM_SiStripCommissioningSources_Averages_H
