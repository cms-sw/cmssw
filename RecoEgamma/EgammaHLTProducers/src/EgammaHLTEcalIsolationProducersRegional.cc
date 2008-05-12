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


// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"


#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"


#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"


EgammaHLTEcalIsolationProducersRegional::EgammaHLTEcalIsolationProducersRegional(const edm::ParameterSet& config) : conf_(config)
{

 // use configuration file to setup input/output collection names

  bcBarrelProducer_           = conf_.getParameter<edm::InputTag>("bcBarrelProducer");
  bcEndcapProducer_           = conf_.getParameter<edm::InputTag>("bcEndcapProducer");

  scIslandBarrelProducer_       = conf_.getParameter<edm::InputTag>("scIslandBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<edm::InputTag>("scIslandEndcapProducer");

  recoEcalCandidateProducer_    = conf_.getParameter<edm::InputTag>("recoEcalCandidateProducer");

  egEcalIsoEtMin_       = conf_.getParameter<double>("egEcalIsoEtMin");
  egEcalIsoConeSize_    = conf_.getParameter<double>("egEcalIsoConeSize");
  algoType_ = conf_.getParameter<int>("SCAlgoType");
  test_ = new EgammaHLTEcalIsolation(egEcalIsoEtMin_,egEcalIsoConeSize_,algoType_);


  //register your products
  produces < reco::RecoEcalCandidateIsolationMap >();
}

EgammaHLTEcalIsolationProducersRegional::~EgammaHLTEcalIsolationProducersRegional(){delete test_;}

// ------------ method called to produce the data  ------------

void EgammaHLTEcalIsolationProducersRegional::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

  // Get the basic cluster collection in the Barrel
  edm::Handle<reco::BasicClusterCollection> bcBarrelHandle;
  iEvent.getByLabel(bcBarrelProducer_, bcBarrelHandle);
  const reco::BasicClusterCollection* clusterBarrelCollection = bcBarrelHandle.product();
  // Get the basic cluster collection in the endcap
  edm::Handle<reco::BasicClusterCollection> bcEndcapHandle;
  iEvent.getByLabel(bcEndcapProducer_, bcEndcapHandle);
  const reco::BasicClusterCollection* clusterEndcapCollection = (bcEndcapHandle.product());
  // Get the  Barrel Super Cluster collection
  edm::Handle<reco::SuperClusterCollection> scBarrelHandle;
  iEvent.getByLabel(scIslandBarrelProducer_,scBarrelHandle);
  const reco::SuperClusterCollection* scBarrelCollection = (scBarrelHandle.product());
  // Get the  Endcap Super Cluster collection
  edm::Handle<reco::SuperClusterCollection> scEndcapHandle;
  iEvent.getByLabel(scIslandEndcapProducer_,scEndcapHandle);
  const reco::SuperClusterCollection* scEndcapCollection = (scEndcapHandle.product());
  // Get the RecoEcalCandidate Collection
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcandHandle;
  iEvent.getByLabel(recoEcalCandidateProducer_,recoecalcandHandle);

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

//define this as a plug-in
//DEFINE_FWK_MODULE(EgammaHLTEcalIsolationProducersRegional);
