#ifndef EventFilter_HGCalRawToDigi_HGCalECONDEmulatorParameters_h
#define EventFilter_HGCalRawToDigi_HGCalECONDEmulatorParameters_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>

namespace hgcal::econd {
  struct EmulatorParameters {
    explicit EmulatorParameters(const edm::ParameterSet&);

    static edm::ParameterSetDescription description();

    const double chan_surv_prob;
    const bool active;
    const bool passthrough_mode;
    const bool expected_mode;
    const bool characterisation_mode;
    const bool matching_ebo_numbers;
    const bool bo_truncated;
    const std::vector<unsigned int> enabled_erxs{};
    const unsigned int header_marker;
    const unsigned int num_channels_per_erx;
    const bool add_econd_crc;
    const bool add_idle_word;
    const unsigned int programmable_pattern;
    struct ErrorProbabilities {
      double bitO{0.}, bitB{0.}, bitE{0.}, bitT{0.}, bitH{0.}, bitS{0.};
    };
    const ErrorProbabilities error_prob;
    const unsigned int default_totstatus;
  };
}  // namespace hgcal::econd

#endif
