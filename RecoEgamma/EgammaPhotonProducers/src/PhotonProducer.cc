#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "RecoEgamma/EgammaPhotonProducers/interface/PhotonProducer.h"


PhotonProducer::PhotonProducer(const edm::ParameterSet& config) : 
  conf_(config) 

{

  // use onfiguration file to setup input/output collection names
  scHybridBarrelProducer_       = conf_.getParameter<std::string>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<std::string>("scIslandEndcapProducer");

  scHybridBarrelCollection_     = conf_.getParameter<std::string>("scHybridBarrelCollection");
  scIslandEndcapCollection_     = conf_.getParameter<std::string>("scIslandEndcapCollection");
  barrelClusterShapeMapProducer_   = conf_.getParameter<std::string>("barrelClusterShapeMapProducer");
  barrelClusterShapeMapCollection_ = conf_.getParameter<std::string>("barrelClusterShapeMapCollection");
  endcapClusterShapeMapProducer_   = conf_.getParameter<std::string>("endcapClusterShapeMapProducer");
  endcapClusterShapeMapCollection_ = conf_.getParameter<std::string>("endcapClusterShapeMapCollection");
  barrelHitProducer_   = conf_.getParameter<std::string>("barrelHitProducer");
  endcapHitProducer_   = conf_.getParameter<std::string>("endcapHitProducer");
  barrelHitCollection_ = conf_.getParameter<std::string>("barrelHitCollection");
  endcapHitCollection_ = conf_.getParameter<std::string>("endcapHitCollection");
  pixelSeedAssocProducer_ = conf_.getParameter<std::string>("pixelSeedAssocProducer");
  vertexProducer_       = conf_.getParameter<std::string>("primaryVertexProducer");
  PhotonCollection_ = conf_.getParameter<std::string>("photonCollection");

  // Parameters for the position calculation:
  std::map<std::string,double> providedParameters;
  providedParameters.insert(std::make_pair("LogWeighted",conf_.getParameter<bool>("posCalc_logweight")));
  providedParameters.insert(std::make_pair("T0_barl",conf_.getParameter<double>("posCalc_t0_barl")));
  providedParameters.insert(std::make_pair("T0_endc",conf_.getParameter<double>("posCalc_t0_endc")));
  providedParameters.insert(std::make_pair("T0_endcPresh",conf_.getParameter<double>("posCalc_t0_endcPresh")));
  providedParameters.insert(std::make_pair("W0",conf_.getParameter<double>("posCalc_w0")));
  providedParameters.insert(std::make_pair("X0",conf_.getParameter<double>("posCalc_x0")));
  posCalculator_ = PositionCalc(providedParameters);

  // Register the product
  produces< reco::PhotonCollection >(PhotonCollection_);

}

PhotonProducer::~PhotonProducer() {

}


void  PhotonProducer::beginJob (edm::EventSetup const & theEventSetup) {


}


void PhotonProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  reco::PhotonCollection outputPhotonCollection;
  std::auto_ptr< reco::PhotonCollection > outputPhotonCollection_p(new reco::PhotonCollection);

  // Get the  Barrel Super Cluster collection
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scHybridBarrelCollection_,scBarrelHandle);

  reco::SuperClusterCollection scBarrelCollection = *(scBarrelHandle.product());
  edm::LogInfo("PhotonProducer") << " Accessing Barrel SC collection with size : " << scBarrelCollection.size()  << "\n";

 // Get the  Endcap Super Cluster collection
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scIslandEndcapCollection_,scEndcapHandle);

  reco::SuperClusterCollection scEndcapCollection = *(scEndcapHandle.product());
  edm::LogInfo("PhotonProducer") << " Accessing Endcap SC collection with size : " << scEndcapCollection.size()  << "\n";
  
  // Get ClusterShape association maps
  Handle<reco::BasicClusterShapeAssociationCollection> barrelClShpHandle;
  theEvent.getByLabel(barrelClusterShapeMapProducer_, barrelClusterShapeMapCollection_, barrelClShpHandle);
  const reco::BasicClusterShapeAssociationCollection& barrelClShpMap = *barrelClShpHandle;

  Handle<reco::BasicClusterShapeAssociationCollection> endcapClShpHandle;
  theEvent.getByLabel(endcapClusterShapeMapProducer_, endcapClusterShapeMapCollection_, endcapClShpHandle);
  const reco::BasicClusterShapeAssociationCollection& endcapClShpMap = *endcapClShpHandle;

  // Get EcalRecHits
  Handle<EcalRecHitCollection> barrelHitHandle;
  theEvent.getByLabel(barrelHitProducer_, barrelHitCollection_, barrelHitHandle);
  const EcalRecHitCollection *barrelRecHits = barrelHitHandle.product();

  Handle<EcalRecHitCollection> endcapHitHandle;
  theEvent.getByLabel(endcapHitProducer_, endcapHitCollection_, endcapHitHandle);
  const EcalRecHitCollection *endcapRecHits = endcapHitHandle.product();

  // get the geometry and topology from the event setup:
  ESHandle<CaloGeometry> geoHandle;
  theEventSetup.get<IdealGeometryRecord>().get(geoHandle);
  const CaloSubdetectorGeometry *barrelGeometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const CaloSubdetectorGeometry *endcapGeometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  const CaloSubdetectorGeometry *preshowerGeometry = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  // Get association map linking SuperClusters to ElectronPixelSeeds
  Handle<reco::SeedSuperClusterAssociationCollection> barrelPixelSeedAssocHandle;
  theEvent.getByLabel(pixelSeedAssocProducer_, scHybridBarrelProducer_, barrelPixelSeedAssocHandle);
  const reco::SeedSuperClusterAssociationCollection& barrelPixelSeedAssoc = *barrelPixelSeedAssocHandle;

  Handle<reco::SeedSuperClusterAssociationCollection> endcapPixelSeedAssocHandle;
  theEvent.getByLabel(pixelSeedAssocProducer_, scIslandEndcapProducer_, endcapPixelSeedAssocHandle);
  const reco::SeedSuperClusterAssociationCollection& endcapPixelSeedAssoc = *endcapPixelSeedAssocHandle;

  // Get the primary event vertex
  Handle<reco::VertexCollection> vertexHandle;
  reco::VertexCollection vertexCollection;
  if (vertexProducer_ != "") {
    theEvent.getByLabel(vertexProducer_, vertexHandle);
    vertexCollection = *(vertexHandle.product());
  }
  math::XYZPoint vtx(0.,0.,0.);
  if (vertexCollection.size()>0) vtx = vertexCollection.begin()->position();

  edm::LogInfo("PhotonProducer") << "Constructing Photon 4-vectors assuming primary vertex position: " << vtx << std::endl;

  int iSC=0; // index in photon collection
  // Loop over barrel and endcap SC collections and fill the  photon collection
  fillPhotonCollection(scBarrelHandle,barrelClShpMap,barrelGeometry,preshowerGeometry,barrelRecHits,barrelPixelSeedAssoc,vtx,outputPhotonCollection,iSC);
  fillPhotonCollection(scEndcapHandle,endcapClShpMap,endcapGeometry,preshowerGeometry,endcapRecHits,endcapPixelSeedAssoc,vtx,outputPhotonCollection,iSC);

  // put the product in the event
  edm::LogInfo("PhotonProducer") << " Put in the event " << iSC << " Photon Candidates \n";
  outputPhotonCollection_p->assign(outputPhotonCollection.begin(),outputPhotonCollection.end());
  theEvent.put( outputPhotonCollection_p, PhotonCollection_);

}

void PhotonProducer::fillPhotonCollection(
		   const edm::Handle<reco::SuperClusterCollection> & scHandle,
		   const reco::BasicClusterShapeAssociationCollection& clshpMap,
		   const CaloSubdetectorGeometry *geometry,
		   const CaloSubdetectorGeometry *geometryES,
		   const EcalRecHitCollection *hits,
		   const reco::SeedSuperClusterAssociationCollection& pixelSeedAssoc,
		   math::XYZPoint & vtx,
		   reco::PhotonCollection & outputPhotonCollection, int& iSC) {

  reco::SuperClusterCollection scCollection = *(scHandle.product());
  reco::SuperClusterCollection::iterator aClus;
  reco::BasicClusterShapeAssociationCollection::const_iterator seedShpItr;
  reco::SeedSuperClusterAssociationCollection::const_iterator pixelSeedAssocItr;

  int lSC=0; // reset local supercluster index
  for(aClus = scCollection.begin(); aClus != scCollection.end(); aClus++) {

    // compute R9=E3x3/ESC
    seedShpItr = clshpMap.find(aClus->seed());
    assert(seedShpItr != clshpMap.end());
    const reco::ClusterShapeRef& seedShapeRef = seedShpItr->val;
    double r9 = seedShapeRef->e3x3()/(aClus->rawEnergy()+aClus->preshowerEnergy());
    double r19 = seedShapeRef->eMax()/seedShapeRef->e3x3();
    double e5x5 = seedShapeRef->e5x5();

    // recalculate position of seed BasicCluster taking shower depth for unconverted photon
    math::XYZPoint unconvPos = posCalculator_.Calculate_Location(aClus->seed()->getHitsByDetId(),hits,geometry,geometryES);

    // compute position of ECAL shower
    math::XYZPoint caloPosition;
    if (r9>0.93) {
      caloPosition = unconvPos;
    } else {
      caloPosition = aClus->position();
    }

    // does the SuperCluster have a matched pixel seed?
    bool hasSeed = false;
    for(pixelSeedAssocItr = pixelSeedAssoc.begin(); pixelSeedAssocItr != pixelSeedAssoc.end(); pixelSeedAssocItr++) {
      if (fabs(pixelSeedAssocItr->val->eta() - aClus->eta()) < 0.0001 &&
	  fabs(pixelSeedAssocItr->val->phi() - aClus->phi()) < 0.0001) {
	hasSeed=true;
	break;
      }
    }

    // compute momentum vector of photon from primary vertex and cluster position
    math::XYZVector direction = caloPosition - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

    reco::Photon newCandidate(0, p4, unconvPos, r9, r19, e5x5, hasSeed, vtx);

    outputPhotonCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scHandle, lSC));
    outputPhotonCollection[iSC].setSuperCluster(scRef);
 
    iSC++;
    lSC++;

  }

}
