#ifndef  PhotonConversionFinderFromTracks_H
#define  PhotonConversionFinderFromTracks_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

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

#include <sstream>
#include "boost/foreach.hpp"

class PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo{

 public:
  PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo(const edm::ParameterSet &,
	edm::ConsumesCollector && iC);
  ~PhotonConversionTrajectorySeedProducerFromQuadrupletsAlgo();

  void init();
  void clear();

  void analyze(const edm::Event & event, const edm::EventSetup & setup);
  TrajectorySeedCollection* getTrajectorySeedCollection(){return seedCollection;}

 private:

  void loop();
  bool inspect(const TrackingRegion & region);

/*  
  :_conf(conf),seedCollection(0),
   hitsfactoryPSet(conf.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet")),   
   creatorPSet(conf.getParameter<edm::ParameterSet>("SeedCreatorPSet")),
   regfactoryPSet(conf.getParameter<edm::ParameterSet>("RegionFactoryPSet")),
   theClusterCheck(conf.getParameter<edm::ParameterSet>("ClusterCheckPSet")),
   SeedComparitorPSet(conf.getParameter<edm::ParameterSet>("SeedComparitorPSet")),
   QuadCutPSet(conf.getParameter<edm::ParameterSet>("QuadCutPSet")),
   theSilentOnClusterCheck(conf.getParameter<edm::ParameterSet>("ClusterCheckPSet").getUntrackedParameter<bool>("silentClusterCheck",false)){
*/
  //Data Members
  const edm::ParameterSet _conf;

  TrajectorySeedCollection *seedCollection;
  edm::ParameterSet creatorPSet;
  ClusterChecker theClusterCheck;
  edm::ParameterSet SeedComparitorPSet,QuadCutPSet;
  bool theSilentOnClusterCheck;

  std::unique_ptr<CombinedHitQuadrupletGeneratorForPhotonConversion> theHitsGenerator;
  SeedForPhotonConversionFromQuadruplets *theSeedCreator;
  std::unique_ptr<GlobalTrackingRegionProducerFromBeamSpot> theRegionProducer;


  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions; 

  edm::Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  reco::Vertex primaryVertex;
  edm::EDGetTokenT<reco::VertexCollection> 	 token_vertex;

  const edm::EventSetup* myEsetup;
  const edm::Event* myEvent;

  PrintRecoObjects po;

  std::stringstream ss;
 
};
#endif
