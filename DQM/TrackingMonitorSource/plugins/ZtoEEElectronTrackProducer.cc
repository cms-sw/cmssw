#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TLorentzVector.h"

#include "DQM/TrackingMonitorSource/interface/ZtoEEElectronTrackProducer.h"

using namespace std;
using namespace edm;

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
      minInvMass_(ps.getUntrackedParameter<double>("minInvMass", 60)),
      maxInvMass_(ps.getUntrackedParameter<double>("maxInvMass", 120)) {
  produces<reco::TrackCollection>("");
}

ZtoEEElectronTrackProducer::~ZtoEEElectronTrackProducer() {}

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
      if (std::fabs(trkd0) >= maxD0_)
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
      float absiso =
          pfIso.sumChargedHadronPt + std::max(0.0, pfIso.sumNeutralHadronEt + pfIso.sumPhotonEt - 0.5 * pfIso.sumPUPt);
      float eiso = absiso / (ele.pt());
      if (eiso > maxIso_)
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
