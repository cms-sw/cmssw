
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DQMOffline/Trigger/interface/HLTDQMTagAndProbeEff.h"

#include <vector>
#include <string>

template <typename TagType, typename TagCollType, typename ProbeType = TagType, typename ProbeCollType = TagCollType>
class HLTTagAndProbeOfflineSource : public DQMEDAnalyzer {
public:
  explicit HLTTagAndProbeOfflineSource(const edm::ParameterSet&);
  ~HLTTagAndProbeOfflineSource() override = default;
  HLTTagAndProbeOfflineSource(const HLTTagAndProbeOfflineSource&) = delete;
  HLTTagAndProbeOfflineSource& operator=(const HLTTagAndProbeOfflineSource&) = delete;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const& run, edm::EventSetup const& c) override;

private:
  std::vector<HLTDQMTagAndProbeEff<TagType, TagCollType, ProbeType, ProbeCollType> > tagAndProbeEffs_;
};

template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
HLTTagAndProbeOfflineSource<TagType, TagCollType, ProbeType, ProbeCollType>::HLTTagAndProbeOfflineSource(
    const edm::ParameterSet& config) {
  auto histCollConfigs = config.getParameter<std::vector<edm::ParameterSet> >("tagAndProbeCollections");
  for (auto& histCollConfig : histCollConfigs) {
    tagAndProbeEffs_.emplace_back(
        HLTDQMTagAndProbeEff<TagType, TagCollType, ProbeType, ProbeCollType>(histCollConfig, consumesCollector()));
  }
}

template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
void HLTTagAndProbeOfflineSource<TagType, TagCollType, ProbeType, ProbeCollType>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("objs", edm::InputTag(""));
  desc.addVPSet("tagAndProbeCollections",
                HLTDQMTagAndProbeEff<TagType, TagCollType, ProbeType, ProbeCollType>::makePSetDescription(),
                std::vector<edm::ParameterSet>());

  // addDefault must be used here instead of add unless this function is specialized
  // for different sets of template parameter types. Each specialization would need
  // a different module label. Otherwise the generated cfi filenames will conflict
  // for the different plugins.
  descriptions.addDefault(desc);
}

template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
void HLTTagAndProbeOfflineSource<TagType, TagCollType, ProbeType, ProbeCollType>::bookHistograms(
    DQMStore::IBooker& iBooker, const edm::Run& run, const edm::EventSetup& setup) {
  for (auto& tpEff : tagAndProbeEffs_)
    tpEff.bookHists(iBooker);
}

template <typename TagType, typename TagCollType, typename ProbeType, typename ProbeCollType>
void HLTTagAndProbeOfflineSource<TagType, TagCollType, ProbeType, ProbeCollType>::analyze(
    const edm::Event& event, const edm::EventSetup& setup) {
  for (auto& tpEff : tagAndProbeEffs_)
    tpEff.fill(event, setup);
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
using HLTEleTagAndProbeOfflineSource = HLTTagAndProbeOfflineSource<reco::GsfElectron, reco::GsfElectronCollection>;
using HLTPhoTagAndProbeOfflineSource = HLTTagAndProbeOfflineSource<reco::Photon, reco::PhotonCollection>;
using HLTElePhoTagAndProbeOfflineSource =
    HLTTagAndProbeOfflineSource<reco::GsfElectron, reco::GsfElectronCollection, reco::Photon, reco::PhotonCollection>;
using HLTMuEleTagAndProbeOfflineSource =
    HLTTagAndProbeOfflineSource<reco::Muon, reco::MuonCollection, reco::GsfElectron, reco::GsfElectronCollection>;
using HLTMuPhoTagAndProbeOfflineSource =
    HLTTagAndProbeOfflineSource<reco::Muon, reco::MuonCollection, reco::Photon, reco::PhotonCollection>;
using HLTMuTagAndProbeOfflineSource = HLTTagAndProbeOfflineSource<reco::Muon, reco::MuonCollection>;
DEFINE_FWK_MODULE(HLTEleTagAndProbeOfflineSource);
DEFINE_FWK_MODULE(HLTPhoTagAndProbeOfflineSource);
DEFINE_FWK_MODULE(HLTElePhoTagAndProbeOfflineSource);
DEFINE_FWK_MODULE(HLTMuEleTagAndProbeOfflineSource);
DEFINE_FWK_MODULE(HLTMuPhoTagAndProbeOfflineSource);
DEFINE_FWK_MODULE(HLTMuTagAndProbeOfflineSource);
