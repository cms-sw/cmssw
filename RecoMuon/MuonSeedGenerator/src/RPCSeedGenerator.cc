// -*- C++ -*-
//
// Package:    RPCSeedGenerator
// Class:      RPCSeedGenerator
//
/**\class RPCSeedGenerator RPCSeedGenerator.cc RecoMuon/MuonSeedGenerator/src/RPCSeedGenerator.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Haiyun Teng
//         Created:  Wed Oct 29 17:24:36 CET 2008
//
//

// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// special include files
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include <vector>
// Using other classes
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedPattern.h"
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedrecHitFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/RPCCosmicSeedrecHitFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedLayerFinder.h"
#include "RecoMuon/MuonSeedGenerator/src/RPCSeedOverlapper.h"
// Geometry
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
// Math
#include <cmath>
// C++
#include <vector>

//
// constants, enums and typedefs
//
using namespace std;
using namespace edm;

typedef MuonTransientTrackingRecHit::MuonRecHitPointer MuonRecHitPointer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
typedef MuonTransientTrackingRecHit::ConstMuonRecHitContainer ConstMuonRecHitContainer;
typedef RPCSeedPattern::weightedTrajectorySeed weightedTrajectorySeed;

#ifndef RPCLayerNumber
#define RPCLayerNumber 12
#endif

#ifndef BarrelLayerNumber
#define BarrelLayerNumber 6
#endif

#ifndef EachEndcapLayerNumber
#define EachEndcapLayerNumber 3
#endif

//
// class decleration
//

class RPCSeedFinder;

class RPCSeedGenerator : public edm::stream::EDProducer<> {
public:
  explicit RPCSeedGenerator(const edm::ParameterSet& iConfig);
  ~RPCSeedGenerator() override;

private:
  void beginStream(edm::StreamID) override;
  void beginRun(const edm::Run&, const edm::EventSetup& iSetup) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void endStream() override;

  // ----------member data ---------------------------
  RPCSeedFinder Finder;
  RPCSeedrecHitFinder recHitFinder;
  RPCCosmicSeedrecHitFinder CosmicrecHitFinder;
  RPCSeedLayerFinder LayerFinder;
  RPCSeedOverlapper Overlapper;
  std::vector<weightedTrajectorySeed> candidateweightedSeeds;
  std::vector<weightedTrajectorySeed> goodweightedSeeds;
  edm::InputTag theRPCRecHits;
  std::unique_ptr<MuonDetLayerMeasurements> muonMeasurements;

  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> brRPCGeometryToken;
  const edm::ESGetToken<MuonDetLayerGeometry, MuonRecoGeometryRecord> muonLayersToken;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeometryToken;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RPCSeedGenerator::RPCSeedGenerator(const edm::ParameterSet& iConfig)
    : brRPCGeometryToken(esConsumes<edm::Transition::BeginRun>()),
      muonLayersToken(esConsumes()),
      fieldToken(esConsumes()),
      rpcGeometryToken(esConsumes()) {
  //register your products
  /* Examples
       produces<ExampleData2>();

    //if do put with a label
    produces<ExampleData2>("label");
    */

  // Now do what ever other initialization is needed
  // Configure other modules
  Finder.configure(iConfig);
  recHitFinder.configure(iConfig);
  CosmicrecHitFinder.configure(iConfig);
  LayerFinder.configure(iConfig);
  Overlapper.configure(iConfig);
  // Register the production
  produces<TrajectorySeedCollection>("goodSeeds");
  produces<TrajectorySeedCollection>("candidateSeeds");
  // Get event data Tag
  theRPCRecHits = iConfig.getParameter<edm::InputTag>("RPCRecHitsLabel");

  // Get RPC recHits by MuonDetLayerMeasurements, while CSC and DT is set to false and with empty InputTag
  edm::ConsumesCollector iC = consumesCollector();

  muonMeasurements = std::make_unique<MuonDetLayerMeasurements>(edm::InputTag(),
                                                                edm::InputTag(),
                                                                theRPCRecHits,
                                                                edm::InputTag(),
                                                                edm::InputTag(),
                                                                iC,
                                                                false,
                                                                false,
                                                                true,
                                                                false,
                                                                false);

  cout << endl << "[RPCSeedGenerator] --> Constructor called" << endl;
}

RPCSeedGenerator::~RPCSeedGenerator() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  cout << "[RPCSeedGenerator] --> Destructor called" << endl;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void RPCSeedGenerator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  goodweightedSeeds.clear();
  candidateweightedSeeds.clear();

  // Create the pointer to the Seed container
  auto goodCollection = std::make_unique<TrajectorySeedCollection>();
  auto candidateCollection = std::make_unique<TrajectorySeedCollection>();

  // Muon Geometry - DT, CSC and RPC
  MuonDetLayerGeometry const& muonLayers = iSetup.getData(muonLayersToken);

  // Get the RPC layers
  vector<const DetLayer*> RPCBarrelLayers = muonLayers.barrelRPCLayers();
  const DetLayer* RB4L = RPCBarrelLayers[5];
  const DetLayer* RB3L = RPCBarrelLayers[4];
  const DetLayer* RB22L = RPCBarrelLayers[3];
  const DetLayer* RB21L = RPCBarrelLayers[2];
  const DetLayer* RB12L = RPCBarrelLayers[1];
  const DetLayer* RB11L = RPCBarrelLayers[0];
  vector<const DetLayer*> RPCEndcapLayers = muonLayers.endcapRPCLayers();
  const DetLayer* REM3L = RPCEndcapLayers[0];
  const DetLayer* REM2L = RPCEndcapLayers[1];
  const DetLayer* REM1L = RPCEndcapLayers[2];
  const DetLayer* REP1L = RPCEndcapLayers[3];
  const DetLayer* REP2L = RPCEndcapLayers[4];
  const DetLayer* REP3L = RPCEndcapLayers[5];

  // Dispatch RPC recHits to the corresponding DetLayer, 6 layers for barrel and 3 layers for each endcap
  MuonRecHitContainer recHitsRPC[RPCLayerNumber];
  recHitsRPC[0] = muonMeasurements->recHits(RB11L, iEvent);
  recHitsRPC[1] = muonMeasurements->recHits(RB12L, iEvent);
  recHitsRPC[2] = muonMeasurements->recHits(RB21L, iEvent);
  recHitsRPC[3] = muonMeasurements->recHits(RB22L, iEvent);
  recHitsRPC[4] = muonMeasurements->recHits(RB3L, iEvent);
  recHitsRPC[5] = muonMeasurements->recHits(RB4L, iEvent);
  recHitsRPC[6] = muonMeasurements->recHits(REM1L, iEvent);
  recHitsRPC[7] = muonMeasurements->recHits(REM2L, iEvent);
  recHitsRPC[8] = muonMeasurements->recHits(REM3L, iEvent);
  recHitsRPC[9] = muonMeasurements->recHits(REP1L, iEvent);
  recHitsRPC[10] = muonMeasurements->recHits(REP2L, iEvent);
  recHitsRPC[11] = muonMeasurements->recHits(REP3L, iEvent);

  // Print the size of recHits in each DetLayer
  cout << "RB1in " << recHitsRPC[0].size() << " recHits" << endl;
  cout << "RB1out " << recHitsRPC[1].size() << " recHits" << endl;
  cout << "RB2in " << recHitsRPC[2].size() << " recHits" << endl;
  cout << "RB2out " << recHitsRPC[3].size() << " recHits" << endl;
  cout << "RB3 " << recHitsRPC[4].size() << " recHits" << endl;
  cout << "RB4 " << recHitsRPC[5].size() << " recHits" << endl;
  cout << "REM1 " << recHitsRPC[6].size() << " recHits" << endl;
  cout << "REM2 " << recHitsRPC[7].size() << " recHits" << endl;
  cout << "REM3 " << recHitsRPC[8].size() << " recHits" << endl;
  cout << "REP1 " << recHitsRPC[9].size() << " recHits" << endl;
  cout << "REP2 " << recHitsRPC[10].size() << " recHits" << endl;
  cout << "REP3 " << recHitsRPC[11].size() << " recHits" << endl;

  // Set Input of RPCSeedFinder, PCSeedrecHitFinder, CosmicrecHitFinder, RPCSeedLayerFinder
  recHitFinder.setInput(recHitsRPC);
  CosmicrecHitFinder.setInput(recHitsRPC);
  LayerFinder.setInput(recHitsRPC);

  // Set Magnetic Field EventSetup of RPCSeedFinder
  const MagneticField& Field = iSetup.getData(fieldToken);

  const RPCGeometry& rpcGeom = iSetup.getData(rpcGeometryToken);
  Finder.setEventSetup(Field, rpcGeom);

  // Start from filling layers to filling seeds
  LayerFinder.fill();
  Overlapper.run();

  // Save seeds to event
  for (vector<weightedTrajectorySeed>::iterator weightedseed = goodweightedSeeds.begin();
       weightedseed != goodweightedSeeds.end();
       ++weightedseed)
    goodCollection->push_back((*weightedseed).first);
  for (vector<weightedTrajectorySeed>::iterator weightedseed = candidateweightedSeeds.begin();
       weightedseed != candidateweightedSeeds.end();
       ++weightedseed)
    candidateCollection->push_back((*weightedseed).first);

  // Put the seed to event
  iEvent.put(std::move(goodCollection), "goodSeeds");
  iEvent.put(std::move(candidateCollection), "candidateSeeds");

  // Unset the input of RPCSeedFinder, PCSeedrecHitFinder, RPCSeedLayerFinder
  recHitFinder.unsetInput();
  CosmicrecHitFinder.unsetInput();
  LayerFinder.unsetInput();
}

void RPCSeedGenerator::beginStream(edm::StreamID) {
  // Set link and EventSetup of RPCSeedFinder, PCSeedrecHitFinder, CosmicrecHitFinder, RPCSeedLayerFinder
  cout << "set link and Geometry EventSetup of RPCSeedFinder, RPCSeedrecHitFinder, RPCCosmicSeedrecHitFinder, "
          "RPCSeedLayerFinder and RPCSeedOverlapper"
       << endl;

  Finder.setOutput(&goodweightedSeeds, &candidateweightedSeeds);
  recHitFinder.setOutput(&Finder);
  CosmicrecHitFinder.setOutput(&Finder);
  LayerFinder.setOutput(&recHitFinder, &CosmicrecHitFinder);
}
void RPCSeedGenerator::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  // Get RPCGeometry
  RPCGeometry const& rpcGeometry = iSetup.getData(brRPCGeometryToken);

  CosmicrecHitFinder.setEdge(rpcGeometry);
  Overlapper.setGeometry(rpcGeometry);
  Overlapper.setIO(&goodweightedSeeds, &candidateweightedSeeds);
}

void RPCSeedGenerator::endStream() { cout << "All jobs completed" << endl; }

//define this as a plug-in
DEFINE_FWK_MODULE(RPCSeedGenerator);
