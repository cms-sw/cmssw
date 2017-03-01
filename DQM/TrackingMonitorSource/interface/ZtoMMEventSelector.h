#ifndef DQM_TrackingMonitorSource_ZtoMMEventSelector_h
#define DQM_TrackingMonitorSource_ZtoMMEventSelector_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class ZtoMMEventSelector : public edm::stream::EDFilter<> {
public:
  explicit ZtoMMEventSelector(const edm::ParameterSet&);

  bool filter(edm::Event&, edm::EventSetup const&) override;

private:
  bool verbose_;
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
