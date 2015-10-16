#ifndef  PhotonConversionFinderFromTracks_H
#define  PhotonConversionFinderFromTracks_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SeedForPhotonConversion1Leg.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"

#include "IdealHelixParameters.h"

#include "PrintRecoObjects.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "CombinedHitPairGeneratorForPhotonConversion.h"

#include "RecoTracker/TkSeedGenerator/interface/ClusterChecker.h"
#include "RecoTracker/TkTrackingRegions/plugins/GlobalTrackingRegionProducerFromBeamSpot.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <sstream>

inline bool lt_(std::pair<double,short> a, std::pair<double,short> b) { return a.first<b.first; }

class dso_hidden PhotonConversionTrajectorySeedProducerFromSingleLegAlgo{

 
 public:
  
  PhotonConversionTrajectorySeedProducerFromSingleLegAlgo(const edm::ParameterSet &,
	edm::ConsumesCollector && iC);
  ~PhotonConversionTrajectorySeedProducerFromSingleLegAlgo();

  void find(const edm::Event & event, const edm::EventSetup & setup, TrajectorySeedCollection & output);

  IdealHelixParameters* getIdealHelixParameters(){return &_IdealHelixParameters;}

 private:

  void loopOnTracks();
  bool inspectTrack(const reco::Track* track, const TrackingRegion & region, math::XYZPoint& primaryVertexPoint);

  bool rejectTrack(const reco::Track& track);

  bool selectPriVtxCompatibleWithTrack(const reco::Track& tk, std::vector<reco::Vertex>& selectedPriVtxCompatibleWithTrack);
  void loopOnPriVtx(const reco::Track& tk, const std::vector<reco::Vertex>& selectedPriVtxCompatibleWithTrack);

  //Data Members

  TrajectorySeedCollection * seedCollection=nullptr;

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

  typedef std::vector<std::unique_ptr<TrackingRegion> > Regions;
  typedef Regions::const_iterator IR;
  Regions regions; 

  edm::Handle<reco::TrackCollection> trackCollectionH;

  const edm::EventSetup* myEsetup;
  const edm::Event* myEvent;

  const MagneticField* magField;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  const reco::BeamSpot * theBeamSpot;

  IdealHelixParameters _IdealHelixParameters;

  edm::Handle<reco::VertexCollection> vertexHandle;
  reco::Vertex primaryVertex;

  PrintRecoObjects po;

  std::stringstream ss;
 
};
#endif
