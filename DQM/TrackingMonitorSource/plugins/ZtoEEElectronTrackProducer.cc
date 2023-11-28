// system includes
#include <memory>

// user includes
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

// ROOT includes
#include "TLorentzVector.h"

class ZtoEEElectronTrackProducer : public edm::global::EDProducer<> {
public:
  explicit ZtoEEElectronTrackProducer(const edm::ParameterSet&);
  ~ZtoEEElectronTrackProducer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID streamID, edm::Event& iEvent, edm::EventSetup const& iSetup) const override;

private:
  // ----------member data ---------------------------
  const edm::InputTag electronTag_;
  const edm::InputTag bsTag_;
  const edm::EDGetTokenT<reco::GsfElectronCollection> electronToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;

  const double maxEta_;
  const double minPt_;
  const double maxDeltaPhiInEB_;
  const double maxDeltaEtaInEB_;
  const double maxHOEEB_;
  const double maxSigmaiEiEEB_;
  const double maxDeltaPhiInEE_;
  const double maxDeltaEtaInEE_;
  const double maxHOEEE_;
  const double maxSigmaiEiEEE_;
  const double maxNormChi2_;
  const double maxD0_;
  const double maxDz_;
  const int minPixelHits_;
  const int minStripHits_;
  const double maxIso_;
  const double minPtHighest_;
  const double minInvMass_;
  const double maxInvMass_;
};

using namespace std;
using namespace edm;

void ZtoEEElectronTrackProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"));
  desc.addUntracked<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"));
  desc.addUntracked<double>("maxEta", 2.4);
  desc.addUntracked<double>("minPt", 5);
  desc.addUntracked<double>("maxDeltaPhiInEB", 0.15);
  desc.addUntracked<double>("maxDeltaEtaInEB", 0.007);
  desc.addUntracked<double>("maxHOEEB", 0.12);
  desc.addUntracked<double>("maxSigmaiEiEEB", 0.01);
  desc.addUntracked<double>("maxDeltaPhiInEE", 0.1);
  desc.addUntracked<double>("maxDeltaEtaInEE", 0.009);
  desc.addUntracked<double>("maxHOEEB_", .10);
  desc.addUntracked<double>("maxSigmaiEiEEE", 0.03);
  desc.addUntracked<double>("maxNormChi2", 10);
  desc.addUntracked<double>("maxD0", 0.02);
  desc.addUntracked<double>("maxDz", 20.);
  desc.addUntracked<unsigned int>("minPixelHits", 1);
  desc.addUntracked<unsigned int>("minStripHits", 8);
  desc.addUntracked<double>("maxIso", 0.3);
  desc.addUntracked<double>("minPtHighest", 24);
  desc.addUntracked<double>("minInvMass", 75);
  desc.addUntracked<double>("maxInvMass", 105);
  descriptions.addWithDefaultLabel(desc);
}

ZtoEEElectronTrackProducer::ZtoEEElectronTrackProducer(const edm::ParameterSet& ps)
    : electronTag_(ps.getUntrackedParameter<edm::InputTag>("electronInputTag", edm::InputTag("gedGsfElectrons"))),
      bsTag_(ps.getUntrackedParameter<edm::InputTag>("offlineBeamSpot", edm::InputTag("offlineBeamSpot"))),
      electronToken_(consumes<reco::GsfElectronCollection>(electronTag_)),
      bsToken_(consumes<reco::BeamSpot>(bsTag_)),
      maxEta_(ps.getUntrackedParameter<double>("maxEta", 2.4)),
      minPt_(ps.getUntrackedParameter<double>("minPt", 5)),
      maxDeltaPhiInEB_(ps.getUntrackedParameter<double>("maxDeltaPhiInEB", 0.15)),
      maxDeltaEtaInEB_(ps.getUntrackedParameter<double>("maxDeltaEtaInEB", 0.007)),
      maxHOEEB_(ps.getUntrackedParameter<double>("maxHOEEB", 0.12)),
      maxSigmaiEiEEB_(ps.getUntrackedParameter<double>("maxSigmaiEiEEB", 0.01)),
      maxDeltaPhiInEE_(ps.getUntrackedParameter<double>("maxDeltaPhiInEE", 0.1)),
      maxDeltaEtaInEE_(ps.getUntrackedParameter<double>("maxDeltaEtaInEE", 0.009)),
      maxHOEEE_(ps.getUntrackedParameter<double>("maxHOEEB_", .10)),
      maxSigmaiEiEEE_(ps.getUntrackedParameter<double>("maxSigmaiEiEEE", 0.03)),
      maxNormChi2_(ps.getUntrackedParameter<double>("maxNormChi2", 10)),
      maxD0_(ps.getUntrackedParameter<double>("maxD0", 0.02)),
      maxDz_(ps.getUntrackedParameter<double>("maxDz", 20.)),
      minPixelHits_(ps.getUntrackedParameter<uint32_t>("minPixelHits", 1)),
      minStripHits_(ps.getUntrackedParameter<uint32_t>("minStripHits", 8)),
      maxIso_(ps.getUntrackedParameter<double>("maxIso", 0.3)),
      minPtHighest_(ps.getUntrackedParameter<double>("minPtHighest", 24)),
      minInvMass_(ps.getUntrackedParameter<double>("minInvMass", 75)),
      maxInvMass_(ps.getUntrackedParameter<double>("maxInvMass", 105)) {
  produces<reco::TrackCollection>("");
}

void ZtoEEElectronTrackProducer::produce(edm::StreamID streamID,
                                         edm::Event& iEvent,
                                         edm::EventSetup const& iSetup) const {
  std::unique_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection());

  // Read Electron Collection
  edm::Handle<reco::GsfElectronCollection> electronColl;
  iEvent.getByToken(electronToken_, electronColl);

  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByToken(bsToken_, beamSpot);

  if (electronColl.isValid()) {
    for (auto const& ele : *electronColl) {
      if (!ele.ecalDriven())
        continue;
      if (ele.pt() < minPt_)
        continue;
      // set a max Eta cut
      if (!(ele.isEB() || ele.isEE()))
        continue;

      double hOverE = ele.hadronicOverEm();
      double sigmaee = ele.sigmaIetaIeta();
      double deltaPhiIn = ele.deltaPhiSuperClusterTrackAtVtx();
      double deltaEtaIn = ele.deltaEtaSuperClusterTrackAtVtx();

      // separate cut for barrel and endcap
      if (ele.isEB()) {
        if (fabs(deltaPhiIn) >= maxDeltaPhiInEB_ && fabs(deltaEtaIn) >= maxDeltaEtaInEB_ && hOverE >= maxHOEEB_ &&
            sigmaee >= maxSigmaiEiEEB_)
          continue;
      } else if (ele.isEE()) {
        if (fabs(deltaPhiIn) >= maxDeltaPhiInEE_ && fabs(deltaEtaIn) >= maxDeltaEtaInEE_ && hOverE >= maxHOEEE_ &&
            sigmaee >= maxSigmaiEiEEE_)
          continue;
      }

      reco::GsfTrackRef trk = ele.gsfTrack();
      reco::TrackRef tk = ele.closestCtfTrackRef();
      if (!trk.isNonnull())
        continue;  // only electrons with tracks
      if (!tk.isNonnull())
        continue;
      double chi2 = trk->chi2();
      double ndof = trk->ndof();
      double chbyndof = (ndof > 0) ? chi2 / ndof : 0;
      if (chbyndof >= maxNormChi2_)
        continue;

      double trkd0 = trk->d0();
      if (beamSpot.isValid()) {
        trkd0 = -(trk->dxy(beamSpot->position()));
      } else {
        edm::LogError("ZtoEEElectronTrackProducer") << "Error >> Failed to get BeamSpot for label: " << bsTag_;
      }
      if (std::abs(trkd0) >= maxD0_)
        continue;

      const reco::HitPattern& hitp = trk->hitPattern();
      int nPixelHits = hitp.numberOfValidPixelHits();
      if (nPixelHits < minPixelHits_)
        continue;

      int nStripHits = hitp.numberOfValidStripHits();
      if (nStripHits < minStripHits_)
        continue;

      // DB corrected PF Isolation
      reco::GsfElectron::PflowIsolationVariables pfIso = ele.pfIsolationVariables();
      const float eiso =
          pfIso.sumChargedHadronPt + std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5 * pfIso.sumPUPt);
      if (eiso > maxIso_ * ele.pt())
        continue;

      outputTColl->push_back(*tk);
    }
  } else {
    edm::LogError("ZtoEEElectronTrackProducer")
        << "Error >> Failed to get ElectronCollection for label: " << electronTag_;
  }

  iEvent.put(std::move(outputTColl));
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ZtoEEElectronTrackProducer);
