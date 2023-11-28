#ifndef L1Trigger_Phase2L1GT_L1GTSingleInOutLUT_h
#define L1Trigger_Phase2L1GT_L1GTSingleInOutLUT_h

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <cinttypes>
#include <cmath>

namespace l1t {

  class L1GTSingleInOutLUT {
  public:
    L1GTSingleInOutLUT(const edm::ParameterSet& lutConfig)
        : data_(lutConfig.getParameter<std::vector<int>>("lut")),
          unused_lsbs_(lutConfig.getParameter<uint32_t>("unused_lsbs")),
          output_scale_(lutConfig.getParameter<double>("output_scale_factor")),
          // I guess ceil is required due to small differences in C++ and python's cos/cosh implementation.
          hwMax_error_(std::ceil(lutConfig.getParameter<double>("max_error") * output_scale_)) {}

    int32_t operator[](uint32_t i) const { return data_[(i >> unused_lsbs_) % data_.size()]; }
    double hwMax_error() const { return hwMax_error_; }
    double output_scale() const { return output_scale_; }

    static void fillLUTDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<std::vector<int32_t>>("lut");
      desc.add<double>("output_scale_factor");
      desc.add<uint32_t>("unused_lsbs");
      desc.add<double>("max_error");
    }

  private:
    const std::vector<int32_t> data_;
    const uint32_t unused_lsbs_;
    const double output_scale_;
    const double hwMax_error_;  // Sanity check
  };
}  // namespace l1t

#endif  // L1Trigger_Phase2L1GT_L1GTSingleInOutLUT_h