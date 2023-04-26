#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TLorentzVector.h"

// Electron selector
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
//#include "RecoEgamma/ElectronIdentification/interface/CutBasedElectronID.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

// Muon Selector
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

// Jet Selector
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

// b-jet Selector
#include "DataFormats/BTauReco/interface/JetTag.h"

// Met Selector
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TH1.h"
#include "DQM/TrackingMonitorSource/interface/ttbarEventSelector.h"

#include "DQM/TrackingMonitorSource/interface/TtbarTrackProducer.h"

using namespace std;
using namespace edm;
//using namespace reco;

TtbarTrackProducer::TtbarTrackProducer(const edm::ParameterSet& ps)
    : electronTag_(ps.getUntrackedParameter<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"))),
      jetsTag_(ps.getUntrackedParameter<edm::InputTag>("jetsInputTag", edm::InputTag("ak4PFJetsCHS"))),  //
      //  bJetsTag_(ps.getUntrackedParameter<edm::InputTag>("jetsInputTag", edm\|#include "DataFormats/EgammaCandidates/interface/Electron.h"
      //						    ::InputTag("slimmedJets"))),

      bjetsTag_(ps.getUntrackedParameter<edm::InputTag>("bjetsInputTag", edm::InputTag("pfDeepCSVJetTags", "probb"))),
      pfmetTag_(ps.getUntrackedParameter<edm::InputTag>("pfmetTag", edm::InputTag("pfMet"))),  //("pfMetT1T2Txy"))),
      muonTag_(ps.getUntrackedParameter<edm::InputTag>("muonInputTag", edm::InputTag("muons"))),
      bsTag_(ps.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"))),
      electronToken_(consumes<reco::GsfElectronCollection>(electronTag_)),
      jetsToken_(consumes<reco::PFJetCollection>(jetsTag_)),
      bjetsToken_(consumes<reco::JetTagCollection>(bjetsTag_)),
      pfmetToken_(consumes<reco::PFMETCollection>(pfmetTag_)),
      muonToken_(consumes<reco::MuonCollection>(muonTag_)),
      bsToken_(consumes<reco::BeamSpot>(bsTag_)),
      maxEtaEle_(ps.getUntrackedParameter<double>("maxEta", 2.4)),
      maxEtaMu_(ps.getUntrackedParameter<double>("maxEta", 2.1)),
      minPt_(ps.getUntrackedParameter<double>("minPt", 5)),

      // for Electron only
      maxDeltaPhiInEB_(ps.getUntrackedParameter<double>("maxDeltaPhiInEB", .15)),
      maxDeltaEtaInEB_(ps.getUntrackedParameter<double>("maxDeltaEtaInEB", .007)),
      maxHOEEB_(ps.getUntrackedParameter<double>("maxHOEEB", .12)),
      maxSigmaiEiEEB_(ps.getUntrackedParameter<double>("maxSigmaiEiEEB", .01)),
      maxDeltaPhiInEE_(ps.getUntrackedParameter<double>("maxDeltaPhiInEE", .1)),
      maxDeltaEtaInEE_(ps.getUntrackedParameter<double>("maxDeltaEtaInEE", .009)),
      maxHOEEE_(ps.getUntrackedParameter<double>("maxHOEEB_", .10)),
      maxSigmaiEiEEE_(ps.getUntrackedParameter<double>("maxSigmaiEiEEE", .03)),

      // for Muon only
      minChambers_(ps.getUntrackedParameter<uint32_t>("minChambers", 2)),
      minMatches_(ps.getUntrackedParameter<uint32_t>("minMatches", 2)),
      minMatchedStations_(ps.getUntrackedParameter<double>("minMatchedStations", 2)),

      // for Jets only
      minEtaHighest_Jets_(ps.getUntrackedParameter<double>("minEtaHighest_Jets", 2.4)),
      minEta_Jets_(ps.getUntrackedParameter<double>("minEta_Jets", 3.0)),

      // for b-tag only
      btagFactor_(ps.getUntrackedParameter<double>("btagFactor", 0.6)),

      maxNormChi2_(ps.getUntrackedParameter<double>("maxNormChi2", 10)),
      maxD0_(ps.getUntrackedParameter<double>("maxD0", 0.02)),
      maxDz_(ps.getUntrackedParameter<double>("maxDz", 20.)),
      minPixelHits_(ps.getUntrackedParameter<uint32_t>("minPixelHits", 1)),
      minStripHits_(ps.getUntrackedParameter<uint32_t>("minStripHits", 8)),
      maxIsoEle_(ps.getUntrackedParameter<double>("maxIso", 0.5)),
      maxIsoMu_(ps.getUntrackedParameter<double>("maxIso", 0.3)),
      minPtHighestMu_(ps.getUntrackedParameter<double>("minPtHighestMu", 24)),
      minPtHighestEle_(ps.getUntrackedParameter<double>("minPtHighestEle", 32)),
      minPtHighest_Jets_(ps.getUntrackedParameter<double>("minPtHighest_Jets", 30)),
      minPt_Jets_(ps.getUntrackedParameter<double>("minPt_Jets", 20)),
      minInvMass_(ps.getUntrackedParameter<double>("minInvMass", 140)),
      maxInvMass_(ps.getUntrackedParameter<double>("maxInvMass", 200)),
      minMet_(ps.getUntrackedParameter<double>("minMet", 50)),
      maxMet_(ps.getUntrackedParameter<double>("maxMet", 80)),
      minWmass_(ps.getUntrackedParameter<double>("minWmass", 50)),
      maxWmass_(ps.getUntrackedParameter<double>("maxWmass", 130)) {
  produces<reco::TrackCollection>("");
}

TtbarTrackProducer::~TtbarTrackProducer() {}

void TtbarTrackProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  std::unique_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection());

  // beamspot
  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);

  // Read Electron Collection
  edm::Handle<reco::GsfElectronCollection> electronColl;
  iEvent.getByToken(electronToken_, electronColl);
  std::vector<TLorentzVector> list_ele;
  std::vector<int> chrgeList_ele;

  // for Track jet collections
  edm::Handle<reco::PFJetCollection> jetColl;
  iEvent.getByToken(jetsToken_, jetColl);
  reco::PFJetCollection Jets;
  std::vector<TLorentzVector> list_jets;

  if (jetColl.isValid()) {
    for (const auto& jets : *jetColl) {
      if (jets.pt() < minPt_Jets_)
        continue;
      if (std::fabs(jets.eta()) < minEta_Jets_)
        continue;
      TLorentzVector lv_jets;  // lv_bJets;
      lv_jets.SetPtEtaPhiE(jets.pt(), jets.eta(), jets.phi(), jets.energy());
      list_jets.push_back(lv_jets);
      Jets.push_back(jets);

      //      if (jets.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags") > btagFactor_) list_bJets.push_back(lv_bJets);

      //  lbj = list_bJets.size();
    }
  }

  edm::Handle<reco::JetTagCollection> bTagHandle;
  iEvent.getByToken(bjetsToken_, bTagHandle);
  const reco::JetTagCollection& bTags = *(bTagHandle.product());
  std::vector<TLorentzVector> list_bjets;
  std::cout << "nbjets " << bTags.size() << std::endl;

  if (!bTags.empty()) {
    for (unsigned bj = 0; bj != bTags.size(); ++bj) {
      TLorentzVector lv_bjets;
      lv_bjets.SetPtEtaPhiE(
          bTags[bj].first->pt(), bTags[bj].first->eta(), bTags[bj].first->phi(), bTags[bj].first->energy());
      if (bTags[bj].second > btagFactor_)
        list_bjets.push_back(lv_bjets);
    }
  }

  for (unsigned int i = 0; i != Jets.size(); i++) {
    reco::TrackRefVector vector = Jets[i].getTrackRefs();
    for (unsigned int j = 0; j != vector.size(); j++) {
      outputTColl->push_back(*vector[j]);
    }
  }

  iEvent.put(std::move(outputTColl));
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TtbarTrackProducer);
