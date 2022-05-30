#include "DQM/GEM/plugins/GEMEffByGEMCSCSegmentClient.h"

GEMEffByGEMCSCSegmentClient::GEMEffByGEMCSCSegmentClient(const edm::ParameterSet& ps)
    : GEMDQMEfficiencyClientBase(ps), kFolder_(ps.getUntrackedParameter<std::string>("folder")) {}

void GEMEffByGEMCSCSegmentClient::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // GEMDQMEfficiencyClientBase
  desc.addUntracked<double>("confidenceLevel", 0.683);  // 1-sigma
  desc.addUntracked<std::string>("logCategory", "GEMEffByGEMCSCSegmentClient");

  // GEMEffByGEMCSCSegmentClient
  desc.addUntracked<std::string>("folder", "GEM/Efficiency/GEMCSCSegment");
  descriptions.addWithDefaultLabel(desc);
}

void GEMEffByGEMCSCSegmentClient::dqmEndLuminosityBlock(DQMStore::IBooker& booker,
                                                        DQMStore::IGetter& getter,
                                                        edm::LuminosityBlock const&,
                                                        edm::EventSetup const&) {
  bookEfficiencyAuto(booker, getter, kFolder_);
}
