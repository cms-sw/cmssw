#include "EventFilter/HGCalRawToDigi/interface/HGCalECONDEmulatorParameters.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalRawDataDefinitions.h"

using namespace hgcal::econd;

EmulatorParameters::EmulatorParameters(const edm::ParameterSet& iConfig)
    : chan_surv_prob(iConfig.getParameter<double>("channelSurv")),
      active(iConfig.getParameter<bool>("active")),
      passthrough_mode(iConfig.getParameter<bool>("passthroughMode")),
      expected_mode(iConfig.getParameter<bool>("expectedMode")),
      characterisation_mode(iConfig.getParameter<bool>("characterisationMode")),
      matching_ebo_numbers(iConfig.getParameter<bool>("matchingEBOnumbers")),
      bo_truncated(iConfig.getParameter<bool>("bufferOverflowTruncated")),
      enabled_erxs(iConfig.getParameter<std::vector<unsigned int> >("enabledERxs")),
      header_marker(iConfig.getParameter<unsigned int>("headerMarker")),
      num_channels_per_erx(iConfig.getParameter<unsigned int>("numChannelsPerERx")),
      add_econd_crc(iConfig.getParameter<bool>("addCRC")),
      add_idle_word(iConfig.getParameter<bool>("addIdleWord")),
      programmable_pattern(iConfig.getParameter<unsigned int>("programmablePattern")),
      error_prob({.bitO = iConfig.getParameter<double>("bitOError"),
                  .bitB = iConfig.getParameter<double>("bitBError"),
                  .bitE = iConfig.getParameter<double>("bitEError"),
                  .bitT = iConfig.getParameter<double>("bitTError"),
                  .bitH = iConfig.getParameter<double>("bitHError"),
                  .bitS = iConfig.getParameter<double>("bitSError")}),
      default_totstatus(iConfig.getParameter<unsigned int>("defaultToTStatus")) {}

edm::ParameterSetDescription EmulatorParameters::description() {
  edm::ParameterSetDescription desc;
  desc.add<double>("channelSurv", 1.);
  desc.add<bool>("active", true)->setComment("is the ECON-D activated?");
  desc.add<bool>("passthroughMode", false)->setComment("ECON-D in pass-through mode?");
  desc.add<bool>("expectedMode", false)->setComment("is an Event HDR/TRL expected to be received from the HGCROCs?");
  desc.add<bool>("characterisationMode", false);
  desc.add<unsigned int>("defaultToTStatus", (unsigned int)ToTStatus::AutomaticFull);
  desc.add<bool>("matchingEBOnumbers", true)
      ->setComment(
          "is the transmitted E/B/O (according to mode selected by user) matching the E/B/O value in the ECON-D "
          "L1A FIFO?");
  desc.add<bool>("bufferOverflowTruncated", false)->setComment("is the packet truncated for buffer overflow?");
  {  // list the enabled eRxs in all ECON-Ds
    const unsigned int max_erxs_per_econd = 12;
    std::vector<unsigned int> default_enabled_erxs;
    for (size_t i = 0; i < max_erxs_per_econd; ++i)
      default_enabled_erxs.emplace_back(i);
    desc.add<std::vector<unsigned int> >("enabledERxs", default_enabled_erxs)
        ->setComment("list of channels to be enabled in readout");
  }
  desc.add<unsigned int>("headerMarker", 0x154)->setComment("9b programmable pattern; default is '0xAA' + '0b0'");
  desc.add<unsigned int>("numChannelsPerERx", 37)->setComment("number of channels managed in each ECON-D eRx");
  desc.add<bool>("addCRC", true)->setComment("add the ECON-D CRC word computed from the whole payload");
  desc.add<bool>("addIdleWord", false)->setComment("add an idle word at the end of each event packet");
  desc.add<unsigned int>("programmablePattern", 0xa5a5a5)
      ->setComment("a 24b programmable pattern used by backend to find event packet");
  desc.add<double>("bitOError", 0.)->setComment("probability that the bit-O error is set");
  desc.add<double>("bitBError", 0.)->setComment("probability that the bit-B error is set");
  desc.add<double>("bitEError", 0.)->setComment("probability that the bit-E error is set");
  desc.add<double>("bitTError", 0.)->setComment("probability that the bit-T error is set");
  desc.add<double>("bitHError", 0.)->setComment("probability that the bit-H error is set");
  desc.add<double>("bitSError", 0.)->setComment("probability that the bit-S error is set");
  return desc;
}
