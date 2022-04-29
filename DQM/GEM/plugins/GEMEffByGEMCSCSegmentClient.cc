#include "DQM/GEM/plugins/GEMEffByGEMCSCSegmentClient.h"

GEMEffByGEMCSCSegmentClient::GEMEffByGEMCSCSegmentClient(const edm::ParameterSet& parameter_set)
    : kFolder_(parameter_set.getUntrackedParameter<std::string>("folder")),
      kLogCategory_(parameter_set.getUntrackedParameter<std::string>("logCategory")) {
  eff_calculator_ = std::make_unique<GEMDQMEfficiencyCalculator>();
}

void GEMEffByGEMCSCSegmentClient::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("folder", "GEM/Efficiency/GEMCSCSegment");
  desc.addUntracked<std::string>("logCategory", "GEMEffByGEMCSCSegmentClient");
  descriptions.addWithDefaultLabel(desc);
}

void GEMEffByGEMCSCSegmentClient::dqmEndLuminosityBlock(DQMStore::IBooker& booker,
                                                        DQMStore::IGetter& getter,
                                                        edm::LuminosityBlock const&,
                                                        edm::EventSetup const&) {
  eff_calculator_->drawEfficiency(booker, getter, kFolder_ + "/Efficiency");
  eff_calculator_->drawEfficiency(booker, getter, kFolder_ + "/Misc");
}
