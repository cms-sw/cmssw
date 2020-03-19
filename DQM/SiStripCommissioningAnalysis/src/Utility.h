#ifndef DQM_SiStripCommissioningAnalysis_Utility_H
#define DQM_SiStripCommissioningAnalysis_Utility_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <vector>
#include <cmath>
#include <cstdint>

namespace sistrip {

  class LinearFit {
  public:
    LinearFit();

    ~LinearFit() { ; }

    class Params {
    public:
      uint16_t n_;
      float a_;
      float b_;
      float erra_;
      float errb_;
      Params()
          : n_(sistrip::invalid_),
            a_(sistrip::invalid_),
            b_(sistrip::invalid_),
            erra_(sistrip::invalid_),
            errb_(sistrip::invalid_) {
        ;
      }
      ~Params() { ; }
    };

    void add(const float& value_x, const float& value_y);

    void add(const float& value_x, const float& value_y, const float& error_y);

    void fit(Params& fit_params);

  private:
    std::vector<float> x_;
    std::vector<float> y_;
    std::vector<float> e_;
    float ss_;
    float sx_;
    float sy_;
  };

  class MeanAndStdDev {
  public:
    MeanAndStdDev();

    ~MeanAndStdDev() { ; }

    class Params {
    public:
      float mean_;
      float rms_;
      float median_;
      Params() : mean_(sistrip::invalid_), rms_(sistrip::invalid_), median_(sistrip::invalid_) { ; }
      ~Params() { ; }
    };

    void add(const float& value, const float& error);

    void fit(Params& fit_params);

  private:
    float s_;
    float x_;
    float xx_;
    std::vector<float> vec_;
  };

}  // namespace sistrip

#endif  // DQM_SiStripCommissioningAnalysis_Utility_H
