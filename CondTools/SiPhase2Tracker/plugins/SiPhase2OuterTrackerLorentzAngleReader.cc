// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerLorentzAngleRcd.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/SiPhase2OuterTrackerLorentzAngle.h"

//
//
// class decleration
//
class SiPhase2OuterTrackerLorentzAngleReader : public edm::global::EDAnalyzer<> {
public:
  explicit SiPhase2OuterTrackerLorentzAngleReader(const edm::ParameterSet&);
  ~SiPhase2OuterTrackerLorentzAngleReader() override = default;
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const uint32_t printdebug_;
  const std::string label_;
  const edm::ESGetToken<SiPhase2OuterTrackerLorentzAngle, SiPhase2OuterTrackerLorentzAngleRcd> laToken_;
};

SiPhase2OuterTrackerLorentzAngleReader::SiPhase2OuterTrackerLorentzAngleReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug", 5)),
      label_(iConfig.getUntrackedParameter<std::string>("label", "")),
      laToken_(esConsumes(edm::ESInputTag{"", label_})) {}

void SiPhase2OuterTrackerLorentzAngleReader::analyze(edm::StreamID,
                                                     edm::Event const& iEvent,
                                                     edm::EventSetup const& iSetup) const {
  const auto& lorentzAngles = iSetup.getData(laToken_);
  edm::LogInfo("SiPhase2OuterTrackerLorentzAngleReader")
      << "[SiPhase2OuterTrackerLorentzAngleReader::analyze] End Reading SiPhase2OuterTrackerLorentzAngle with label "
      << label_ << std::endl;

  const auto& detid_la = lorentzAngles.getLorentzAngles();
  std::unordered_map<unsigned int, float>::const_iterator it;
  size_t count = 0;
  for (it = detid_la.begin(); it != detid_la.end() && count < printdebug_; it++) {
    edm::LogInfo("SiPhase2OuterTrackerLorentzAngleReader") << "detid " << it->first << " \t"
                                                           << " Lorentz angle  " << it->second;
    count++;
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiPhase2OuterTrackerLorentzAngleReader::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Module to read SiPhase2OuterTrackerLorentzAngle Payloads");
  desc.addUntracked<uint32_t>("printDebug", 5)->setComment("maximum amount of print-outs");
  desc.addUntracked<std::string>("label", "")->setComment("label from which to read the payload");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPhase2OuterTrackerLorentzAngleReader);
