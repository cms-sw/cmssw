#ifndef RecoMuon_TrackerSeedGenerator_TSGFromL2Muon_H
#define RecoMuon_TrackerSeedGenerator_TSGFromL2Muon_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedCleaner.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <vector>

namespace edm { class ParameterSet; class Event; class EventSetup; }

class MuonServiceProxy;
class TrackerSeedGenerator;
class MuonTrackingRegionBuilder;
class TrackerSeedCleaner;

//
// Generate tracker seeds from L2 muons
//
class TSGFromL2Muon : public edm::stream::EDProducer<> {
    
  public:
    TSGFromL2Muon(const edm::ParameterSet& cfg);
    virtual ~TSGFromL2Muon();
    virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
    virtual void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    edm::ParameterSet theConfig;
    edm::InputTag theL2CollectionLabel;
    MuonServiceProxy* theService;
    double thePtCut,thePCut;
    MuonTrackingRegionBuilder* theRegionBuilder;
    TrackerSeedGenerator* theTkSeedGenerator;
    TrackerSeedCleaner* theSeedCleaner;
    edm::EDGetTokenT<reco::TrackCollection> l2muonToken;
};
#endif
