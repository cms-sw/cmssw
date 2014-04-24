#ifndef  PhotonConversionFinderFromTracks_H
#define  PhotonConversionFinderFromTracks_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/ConversionSeedGenerators/interface/SeedForPhotonConversion1Leg.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"

#include "RecoTracker/ConversionSeedGenerators/interface/IdealHelixParameters.h"

#include "RecoTracker/ConversionSeedGenerators/interface/PrintRecoObjects.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoTracker/ConversionSeedGenerators/interface/CombinedHitPairGeneratorForPhotonConversion.h"

#include "RecoTracker/TkSeedGenerator/interface/ClusterChecker.h"
#include "RecoTracker/TkTrackingRegions/plugins/GlobalTrackingRegionProducerFromBeamSpot.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <sstream>
#include "boost/foreach.hpp"

inline bool lt_(std::pair<double,short> a, std::pair<double,short> b) { return a.first<b.first; }

class PhotonConversionTrajectorySeedProducerFromSingleLegAlgo{

 
 public:
  
  PhotonConversionTrajectorySeedProducerFromSingleLegAlgo(const edm::ParameterSet &,
	edm::ConsumesCollector && iC);
  ~PhotonConversionTrajectorySeedProducerFromSingleLegAlgo();

  void analyze(const edm::Event & event, const edm::EventSetup & setup);
  IdealHelixParameters* getIdealHelixParameters(){return &_IdealHelixParameters;}
  TrajectorySeedCollection* getTrajectorySeedCollection(){return seedCollection;}
  TrajectorySeedCollection* getTrajectorySeedCollectionOfSourceTracks(){return seedCollectionOfSourceTracks;}

 private:

  void loopOnTracks();
  bool inspectTrack(const reco::Track* track, const TrackingRegion & region, math::XYZPoint& primaryVertexPoint);

  bool rejectTrack(const reco::Track& track);

  bool selectPriVtxCompatibleWithTrack(const reco::Track& tk, std::vector<reco::Vertex>& selectedPriVtxCompatibleWithTrack);
  void loopOnPriVtx(const reco::Track& tk, const std::vector<reco::Vertex>& selectedPriVtxCompatibleWithTrack);

  //Data Members
  const edm::ParameterSet _conf;

  TrajectorySeedCollection *seedCollection;
  TrajectorySeedCollection *seedCollectionOfSourceTracks;
  std::unique_ptr<CombinedHitPairGeneratorForPhotonConversion> theHitsGenerator;
  std::unique_ptr<SeedForPhotonConversion1Leg> theSeedCreator;
  std::unique_ptr<GlobalTrackingRegionProducerFromBeamSpot> theRegionProducer;


  ClusterChecker theClusterCheck;
  bool theSilentOnClusterCheck;

  double _vtxMinDoF, _maxDZSigmas;
  size_t _maxNumSelVtx;
  bool   _applyTkVtxConstraint;
  size_t _countSeedTracks;
  edm::InputTag _primaryVtxInputTag, _beamSpotInputTag;
  edm::EDGetTokenT<reco::VertexCollection> token_vertex; 
  edm::EDGetTokenT<reco::BeamSpot> token_bs; 
  edm::EDGetTokenT<reco::TrackCollection> token_refitter; 

  typedef std::vector<TrackingRegion* > Regions;
  typedef Regions::const_iterator IR;
  Regions regions; 

  edm::Handle<TrajTrackAssociationCollection> trajTrackAssociations;
  edm::Handle<reco::TrackCollection> trackCollectionH;

  const edm::EventSetup* myEsetup;
  const edm::Event* myEvent;

  const MagneticField* magField;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  const reco::BeamSpot * theBeamSpot;

  IdealHelixParameters _IdealHelixParameters;

  edm::Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  reco::Vertex primaryVertex;

  PrintRecoObjects po;

  std::stringstream ss;
 
};
#endif
