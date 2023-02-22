#ifndef EventFilter_HGCalRawToDigi_HGCalECONDEmulatorParameters_h
#define EventFilter_HGCalRawToDigi_HGCalECONDEmulatorParameters_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>

namespace hgcal::econd {
  struct EmulatorParameters {
    explicit EmulatorParameters(const edm::ParameterSet& iConfig)
        : chan_surv_prob(iConfig.getParameter<double>("channelSurv")),
          enabled_erxs(iConfig.getParameter<std::vector<unsigned int> >("enabledERxs")),
          header_marker(iConfig.getParameter<unsigned int>("headerMarker")),
          num_channels_per_erx(iConfig.getParameter<unsigned int>("numChannelsPerERx")),
          error_prob({.bitO = iConfig.getParameter<double>("bitOError"),
                      .bitB = iConfig.getParameter<double>("bitBError"),
                      .bitE = iConfig.getParameter<double>("bitEError"),
                      .bitT = iConfig.getParameter<double>("bitTError"),
                      .bitH = iConfig.getParameter<double>("bitHError"),
                      .bitS = iConfig.getParameter<double>("bitSError")}) {}

    static edm::ParameterSetDescription description() {
      edm::ParameterSetDescription desc;
      desc.add<double>("channelSurv", 1.);
      desc.add<std::vector<unsigned int> >("enabledERxs", {})->setComment("list of channels to be enabled in readout");
      desc.add<unsigned int>("headerMarker", 0x154)->setComment("9b programmable pattern; default is '0xAA' + '0b0'");
      desc.add<unsigned int>("numChannelsPerERx", 37)->setComment("number of channels managed in ECON-D");
      desc.add<double>("bitOError", 0.)->setComment("probability that the bit-O error is set");
      desc.add<double>("bitBError", 0.)->setComment("probability that the bit-B error is set");
      desc.add<double>("bitEError", 0.)->setComment("probability that the bit-E error is set");
      desc.add<double>("bitTError", 0.)->setComment("probability that the bit-T error is set");
      desc.add<double>("bitHError", 0.)->setComment("probability that the bit-H error is set");
      desc.add<double>("bitSError", 0.)->setComment("probability that the bit-S error is set");
      return desc;
    }

    const double chan_surv_prob{1.};
    const std::vector<unsigned int> enabled_erxs{};
    const unsigned int header_marker{0};
    const unsigned int num_channels_per_erx{0};
    struct ErrorProbabilities {
      double bitO{0.}, bitB{0.}, bitE{0.}, bitT{0.}, bitH{0.}, bitS{0.};
    };
    const ErrorProbabilities error_prob;
  };
}  // namespace hgcal::econd

#endif
