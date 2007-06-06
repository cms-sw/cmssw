//
// Package:         RecoEgamma/EgammaHLTProducers
// Class:           EgammaHLTRegionalPixelSeedGeneratorProducers
//  Modified from TkSeedGeneratorFromTrk by Jeremy Werner, Princeton University, USA
// $Id: EgammaHLTRegionalPixelSeedGeneratorProducers.cc,v 1.3 2007/05/09 08:15:51 ghezzi Exp $
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
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace std;
using namespace reco;

EgammaHLTRegionalPixelSeedGeneratorProducers::EgammaHLTRegionalPixelSeedGeneratorProducers(edm::ParameterSet const& conf) :   conf_(conf),combinatorialSeedGenerator(conf)
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

}

// Virtual destructor needed.
EgammaHLTRegionalPixelSeedGeneratorProducers::~EgammaHLTRegionalPixelSeedGeneratorProducers() { }  

// Functions that gets called by framework every event
void EgammaHLTRegionalPixelSeedGeneratorProducers::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get Inputs
  edm::Handle<SiPixelRecHitCollection> pixelHits;

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  //
  // get the pixel Hits
  //
  std::string hitProducer = conf_.getParameter<std::string>("HitProducer");
  iEvent.getByLabel(hitProducer, pixelHits);

  std::auto_ptr<TrajectorySeedCollection> output(new TrajectorySeedCollection());    

  //

  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcands;
  iEvent.getByLabel(candTag_,recoecalcands);

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
	  zvertex = iElectron->track()->vz();	  
	  break;
	}
      }

    }
    GlobalVector dirVector((recoecalcand)->px(),(recoecalcand)->py(),(recoecalcand)->pz());
    RectangularEtaPhiTrackingRegion* etaphiRegion = new  RectangularEtaPhiTrackingRegion(dirVector,
											   GlobalPoint(0,0,zvertex), 
											   ptmin_,
											   originradius_,
											   halflength_,
											   deltaEta_,
											   deltaPhi_);

    combinatorialSeedGenerator.init(*pixelHits,iSetup);
    combinatorialSeedGenerator.run(*etaphiRegion,*output,iSetup);
    
    delete etaphiRegion;
    
  }

    iEvent.put(output);
}
