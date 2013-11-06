/** \class EgammaHLTEcalIsolationProducersRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id:
 */

#include <iostream>
#include <vector>
#include <memory>

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTEcalIsolationProducersRegional.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EgammaHLTEcalIsolationProducersRegional::EgammaHLTEcalIsolationProducersRegional(const edm::ParameterSet& config) : conf_(config) {
  
  // use configuration file to setup input/output collection names
  
  bcBarrelProducer_           = consumes<reco::BasicClusterCollection>(conf_.getParameter<edm::InputTag>("bcBarrelProducer"));
  bcEndcapProducer_           = consumes<reco::BasicClusterCollection>(conf_.getParameter<edm::InputTag>("bcEndcapProducer"));
  scIslandBarrelProducer_     = consumes<reco::SuperClusterCollection>(conf_.getParameter<edm::InputTag>("scIslandBarrelProducer"));
  scIslandEndcapProducer_     = consumes<reco::SuperClusterCollection>(conf_.getParameter<edm::InputTag>("scIslandEndcapProducer"));
  recoEcalCandidateProducer_  = consumes<reco::RecoEcalCandidateCollection>(conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer"));

  egEcalIsoEtMin_       = conf_.getParameter<double>("egEcalIsoEtMin");
  egEcalIsoConeSize_    = conf_.getParameter<double>("egEcalIsoConeSize");
  algoType_ = conf_.getParameter<int>("SCAlgoType");
  test_ = new EgammaHLTEcalIsolation(egEcalIsoEtMin_,egEcalIsoConeSize_,algoType_);

  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
}

EgammaHLTEcalIsolationProducersRegional::~EgammaHLTEcalIsolationProducersRegional() {
  delete test_;
}

void EgammaHLTEcalIsolationProducersRegional::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc; 
  desc.add<edm::InputTag>("bcBarrelProducer", edm::InputTag(""));
  desc.add<edm::InputTag>("bcEndcapProducer", edm::InputTag(""));
  desc.add<edm::InputTag>("scIslandBarrelProducer", edm::InputTag(""));
  desc.add<edm::InputTag>("scIslandEndcapProducer", edm::InputTag(""));
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag(""));
  desc.add<double>("egEcalIsoEtMin", 0.);
  desc.add<double>("egEcalIsoConeSize", 0.3);
  desc.add<int>("SCAlgoType", 1);
  descriptions.add("hltEgammaHLTEcalIsolationProducersRegional", desc);  
}

  
void EgammaHLTEcalIsolationProducersRegional::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

  // Get the basic cluster collection in the Barrel
  edm::Handle<reco::BasicClusterCollection> bcBarrelHandle;
  iEvent.getByToken(bcBarrelProducer_, bcBarrelHandle);
  const reco::BasicClusterCollection* clusterBarrelCollection = bcBarrelHandle.product();
  // Get the basic cluster collection in the endcap
  edm::Handle<reco::BasicClusterCollection> bcEndcapHandle;
  iEvent.getByToken(bcEndcapProducer_, bcEndcapHandle);
  const reco::BasicClusterCollection* clusterEndcapCollection = (bcEndcapHandle.product());
  // Get the  Barrel Super Cluster collection
  edm::Handle<reco::SuperClusterCollection> scBarrelHandle;
  iEvent.getByToken(scIslandBarrelProducer_,scBarrelHandle);
  const reco::SuperClusterCollection* scBarrelCollection = (scBarrelHandle.product());
  // Get the  Endcap Super Cluster collection
  edm::Handle<reco::SuperClusterCollection> scEndcapHandle;
  iEvent.getByToken(scIslandEndcapProducer_,scEndcapHandle);
  const reco::SuperClusterCollection* scEndcapCollection = (scEndcapHandle.product());
  // Get the RecoEcalCandidate Collection
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByToken(recoEcalCandidateProducer_,recoecalcandHandle);

  std::vector<const reco::BasicCluster*> clusterCollection;
  for (reco::BasicClusterCollection::const_iterator ibc = clusterBarrelCollection->begin(); 
       ibc < clusterBarrelCollection->end(); ibc++ ){clusterCollection.push_back(&(*ibc));}
  for (reco::BasicClusterCollection::const_iterator iec = clusterEndcapCollection->begin(); 
       iec < clusterEndcapCollection->end(); iec++ ){clusterCollection.push_back(&(*iec));}
  std::vector<const reco::SuperCluster*> scCollection;
  for (reco::SuperClusterCollection::const_iterator ibsc = scBarrelCollection->begin(); 
       ibsc < scBarrelCollection->end(); ibsc++ ){scCollection.push_back(&(*ibsc));}
  for (reco::SuperClusterCollection::const_iterator iesc = scEndcapCollection->begin(); 
       iesc < scEndcapCollection->end(); iesc++ ){scCollection.push_back(&(*iesc));}

  reco::RecoEcalCandidateIsolationMap isoMap;



 for (reco::RecoEcalCandidateCollection::const_iterator iRecoEcalCand= recoecalcandHandle->begin(); iRecoEcalCand!=recoecalcandHandle->end(); iRecoEcalCand++) {


    reco::RecoEcalCandidateRef recoecalcandref(reco::RecoEcalCandidateRef(recoecalcandHandle,iRecoEcalCand -recoecalcandHandle ->begin()));

    
    const reco::RecoCandidate *tempiRecoEcalCand = &(*recoecalcandref);
    float isol =  test_->isolPtSum(tempiRecoEcalCand,scCollection, clusterCollection);

    isoMap.insert(recoecalcandref, isol);

  }

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> isolMap(new reco::RecoEcalCandidateIsolationMap(isoMap));
  iEvent.put(isolMap);

}
