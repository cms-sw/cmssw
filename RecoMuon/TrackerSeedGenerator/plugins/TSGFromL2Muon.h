#ifndef RecoMuon_TrackerSeedGenerator_TSGFromL2Muon_H
#define RecoMuon_TrackerSeedGenerator_TSGFromL2Muon_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>


#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedCleaner.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"



namespace edm { class ParameterSet; class Event; class EventSetup; }

class MuonServiceProxy;
class TrackerSeedGenerator;
class MuonTrackingRegionBuilder;
class TrackerSeedCleaner;

//
// generate seeds corresponding to L2 muons
//

class TSGFromL2Muon : public edm::stream::EDProducer<> {
public:
  TSGFromL2Muon(const edm::ParameterSet& cfg);
  virtual ~TSGFromL2Muon();
  virtual void beginRun(const edm::Run & run, const edm::EventSetup&es) override;
  virtual void produce(edm::Event& ev, const edm::EventSetup& es) override;
 
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
