#ifndef GlobalTrackingTools_GlobalTrackQualityProducer_h
#define GlobalTrackingTools_GlobalTrackQualityProducer_h

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonQuality.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonRefitter.h"
#include "RecoMuon/GlobalTrackingTools/interface/GlobalMuonTrackMatcher.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "TrackingTools/DetLayers/interface/MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

class GlobalMuonRefitter;

class GlobalTrackQualityProducer : public edm::stream::EDProducer<> {
public:
  explicit GlobalTrackQualityProducer(const edm::ParameterSet& iConfig);

  ~GlobalTrackQualityProducer() override;  // {}

  // describe the parameters it allows or requires to be in its configuration
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual std::pair<double, double> kink(Trajectory& muon) const;
  virtual std::pair<double, double> newChi2(Trajectory& muon) const;
  virtual double trackProbability(Trajectory& track) const;

  edm::InputTag inputCollection_;
  edm::InputTag inputLinksCollection_;
  edm::EDGetTokenT<reco::TrackCollection> glbMuonsToken;
  edm::EDGetTokenT<reco::MuonTrackLinksCollection> linkCollectionToken;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;
  MuonServiceProxy* theService;
  GlobalMuonRefitter* theGlbRefitter;
  GlobalMuonTrackMatcher* theGlbMatcher;
  MeasurementEstimator* theEstimator;
  //muon::SelectionType selectionType_;
};
#endif
