#ifndef  PhotonConversionFinderFromTracks_H
#define  PhotonConversionFinderFromTracks_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/ConversionSeedGenerators/interface/SeedForPhotonConversionFromQuadruplets.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"


#include "RecoTracker/ConversionSeedGenerators/interface/PrintRecoObjects.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoTracker/ConversionSeedGenerators/interface/CombinedHitQuadrupletGeneratorForPhotonConversion.h"

#include "RecoTracker/SpecialSeedGenerators/interface/ClusterChecker.h"
#include "RecoTracker/TkTrackingRegions/plugins/GlobalTrackingRegionProducerFromBeamSpot.h"

#include "sstream"
#include "boost/foreach.hpp"

class PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo{

 public:
  PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo(const edm::ParameterSet &);
  ~PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo(){};

  void init();
  void clear();

  void analyze(const edm::Event & event, const edm::EventSetup & setup);
  TrajectorySeedCollection* getTrajectorySeedCollection(){return seedCollection;}

 private:

  void loop();
  bool inspect(const TrackingRegion & region);

  //Data Members
  const edm::ParameterSet _conf;

  TrajectorySeedCollection *seedCollection;
  CombinedHitQuadrupletGeneratorForPhotonConversion * theHitsGenerator;
  SeedForPhotonConversionFromQuadruplets *theSeedCreator;
  GlobalTrackingRegionProducerFromBeamSpot* theRegionProducer;

  edm::ParameterSet hitsfactoryPSet,creatorPSet,regfactoryPSet;

  ClusterChecker theClusterCheck;
  bool theSilentOnClusterCheck;

  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions; 

  edm::Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  reco::Vertex primaryVertex;

  const edm::EventSetup* myEsetup;
  const edm::Event* myEvent;

  PrintRecoObjects po;

  std::stringstream ss;
 
};
#endif
