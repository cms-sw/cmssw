//
// Package:         RecoEgamma/EgammaHLTProducers
// Class:           EgammaHLTRegionalPixelSeedGeneratorProducers
//  Modified from TkSeedGeneratorFromTrk by Jeremy Werner, Princeton University, USA
// $Id: EgammaHLTRegionalPixelSeedGeneratorProducers.cc,v 1.14 2013/02/27 19:53:13 eulisse Exp $
//

#include <iostream>
#include <memory>
#include <string>

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTRegionalPixelSeedGeneratorProducers.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGeneratorFactory.h"
#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedGeneratorFromRegionHits.h"
#include "RecoTracker/TkSeedGenerator/interface/SeedCreatorFactory.h"

#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"//needed?
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"//needed?

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace std;
using namespace reco;

EgammaHLTRegionalPixelSeedGeneratorProducers::EgammaHLTRegionalPixelSeedGeneratorProducers(edm::ParameterSet const& conf) :   conf_(conf),combinatorialSeedGenerator(0)
{

  produces<TrajectorySeedCollection>();

  ptmin_       = conf_.getParameter<double>("ptMin");
  vertexz_     = conf_.getParameter<double>("vertexZ");
  originradius_= conf_.getParameter<double>("originRadius");
  halflength_  = conf_.getParameter<double>("originHalfLength");
  deltaEta_    = conf_.getParameter<double>("deltaEtaRegion");
  deltaPhi_    = conf_.getParameter<double>("deltaPhiRegion");
  candTag_     = conf_.getParameter< edm::InputTag > ("candTag");
  candTagEle_  = conf_.getParameter< edm::InputTag > ("candTagEle");
  useZvertex_  = conf_.getParameter<bool>("UseZInVertex");
  BSProducer_ = conf.getParameter<edm::InputTag>("BSProducer");
  // setup orderedhits setup (in order to tell seed generator to use pairs/triplets, which layers)
}

// Virtual destructor needed.
EgammaHLTRegionalPixelSeedGeneratorProducers::~EgammaHLTRegionalPixelSeedGeneratorProducers() { 
}  

void EgammaHLTRegionalPixelSeedGeneratorProducers::endRun(edm::Run const&run, const edm::EventSetup& es)
{
  delete combinatorialSeedGenerator;
  combinatorialSeedGenerator=0;
}


void EgammaHLTRegionalPixelSeedGeneratorProducers::beginRun(edm::Run const&run, const edm::EventSetup& es)
{
  edm::ParameterSet hitsfactoryPSet =
      conf_.getParameter<edm::ParameterSet>("OrderedHitsFactoryPSet");
  std::string hitsfactoryName = hitsfactoryPSet.getParameter<std::string>("ComponentName");

  // get orderd hits generator from factory
  OrderedHitsGenerator*  hitsGenerator =
        OrderedHitsGeneratorFactory::get()->create( hitsfactoryName, hitsfactoryPSet);

  // start seed generator
  // FIXME??
  edm::ParameterSet creatorPSet;
  creatorPSet.addParameter<std::string>("propagator","PropagatorWithMaterial");

  combinatorialSeedGenerator = new SeedGeneratorFromRegionHits( hitsGenerator, 0, 
						 SeedCreatorFactory::get()->create("SeedFromConsecutiveHitsCreator", creatorPSet)
                                                              );
}

// Functions that gets called by framework every event
void EgammaHLTRegionalPixelSeedGeneratorProducers::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // resulting collection
  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());    

  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcands;
  iEvent.getByLabel(candTag_,recoecalcands);

  //Get the Beam Spot position
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  // iEvent.getByType(recoBeamSpotHandle);
  iEvent.getByLabel(BSProducer_,recoBeamSpotHandle);
  // gets its position
  const BeamSpot::Point& BSPosition = recoBeamSpotHandle->position(); 

  //Get the HLT electrons collection if needed
  edm::Handle<reco::ElectronCollection> electronHandle;
  if(useZvertex_){iEvent.getByLabel(candTagEle_,electronHandle);}

  reco::SuperClusterRef scRef;
  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoecalcands->begin(); recoecalcand!=recoecalcands->end(); recoecalcand++) {
    scRef = recoecalcand->superCluster();
    float zvertex = 0;
    if( useZvertex_ ){
      reco::SuperClusterRef scRefEle;
      for(reco::ElectronCollection::const_iterator iElectron = electronHandle->begin(); iElectron != electronHandle->end(); iElectron++){
	//Compare electron SC with EcalCandidate SC
	scRefEle = iElectron->superCluster();
	if(&(*scRef) == &(*scRefEle)){
	  if(iElectron->track().isNonnull()) zvertex = iElectron->track()->vz();
	  else  zvertex = iElectron->gsfTrack()->vz();
	  break;
	}
      }

    }
    GlobalVector dirVector((recoecalcand)->px(),(recoecalcand)->py(),(recoecalcand)->pz());
    RectangularEtaPhiTrackingRegion etaphiRegion( dirVector,
											   GlobalPoint( BSPosition.x(), BSPosition.y(), zvertex ), 
											   ptmin_,
											   originradius_,
											   halflength_,
											   deltaEta_,
											   deltaPhi_);

    // fill Trajectory seed collection
    combinatorialSeedGenerator->run(*output, etaphiRegion, iEvent, iSetup);
    
  }

    iEvent.put(output);
}
