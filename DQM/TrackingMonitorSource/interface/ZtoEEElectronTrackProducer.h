#ifndef DQM_TrackingMonitorSource_ZtoEEElectronTrackProducer_h
#define DQM_TrackingMonitorSource_ZtoEEElectronTrackProducer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class ZtoEEElectronTrackProducer : public edm::one::EDProducer<> {
public:
  explicit ZtoEEElectronTrackProducer(const edm::ParameterSet&);
  ~ZtoEEElectronTrackProducer() override;

  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

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
#endif
