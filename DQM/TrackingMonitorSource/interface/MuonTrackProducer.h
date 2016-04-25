#ifndef DQM_TrackingMonitorSource_MuonTrackProducer_h
#define DQM_TrackingMonitorSource_MuonTrackProducer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"	

class MuonTrackProducer : public edm::EDProducer {
public:
  explicit MuonTrackProducer (const edm::ParameterSet&);
  ~MuonTrackProducer();

  virtual void produce(edm::Event& iEvent, edm::EventSetup const& iSetup);

private:

      // ----------member data ---------------------------

  const edm::InputTag muonTag_;
  const edm::InputTag bsTag_;
  const edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;

  const double maxEta_;
  const double minPt_;
  const double maxNormChi2_;
  const double maxD0_;
  const double maxDz_;
  const int minPixelHits_;
  const int minStripHits_;
  const int minChambers_;
  const int minMatches_;
  const int minMatchedStations_;
  const double maxIso_;
  const double minPtHighest_;
  const double minInvMass_;
  const double maxInvMass_;
};
#endif
