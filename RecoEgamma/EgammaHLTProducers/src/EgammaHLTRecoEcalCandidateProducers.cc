/** \class EgammaHLTRecoEcalCandidateProducers
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 * $Id:
 *
 */

#include <iostream>
#include <vector>
#include <memory>

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
//
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTRecoEcalCandidateProducers.h"


EgammaHLTRecoEcalCandidateProducers::EgammaHLTRecoEcalCandidateProducers(const edm::ParameterSet& config) : 
  conf_(config) 

{
  // use onfiguration file to setup input/output collection names
  scHybridBarrelProducer_       = conf_.getParameter<edm::InputTag>("scHybridBarrelProducer");
  scIslandEndcapProducer_       = conf_.getParameter<edm::InputTag>("scIslandEndcapProducer");
  recoEcalCandidateCollection_  = conf_.getParameter<std::string>("recoEcalCandidateCollection");

  // Register the product
  produces< reco::RecoEcalCandidateCollection >(recoEcalCandidateCollection_);
}

EgammaHLTRecoEcalCandidateProducers::~EgammaHLTRecoEcalCandidateProducers() {}

void EgammaHLTRecoEcalCandidateProducers::beginJob() {}

void EgammaHLTRecoEcalCandidateProducers::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  using namespace edm;

  //
  // create empty output collections
  //

  reco::RecoEcalCandidateCollection outputRecoEcalCandidateCollection;
  std::auto_ptr< reco::RecoEcalCandidateCollection > outputRecoEcalCandidateCollection_p(new reco::RecoEcalCandidateCollection);

  // Get the  Barrel Super Cluster collection
  Handle<reco::SuperClusterCollection> scBarrelHandle;
  theEvent.getByLabel(scHybridBarrelProducer_,scBarrelHandle);
  // Get the  Endcap Super Cluster collection
  Handle<reco::SuperClusterCollection> scEndcapHandle;
  theEvent.getByLabel(scIslandEndcapProducer_,scEndcapHandle);

  //  Loop over barrel SC and fill the  recoecal collection
  int iSC=0; // index in recoecal collection
  int lSC=0; // local index on barrel


for(reco::SuperClusterCollection::const_iterator aClus = scBarrelHandle->begin(); aClus != scBarrelHandle->end(); aClus++) {

    const reco::Particle::Point  vtx( 0, 0, 0 );

    // compute correctly the momentum vector of the recoecal from primary vertex and cluster position
    math::XYZVector direction =aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

    reco::RecoEcalCandidate newCandidate(0, p4, vtx);

    outputRecoEcalCandidateCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scBarrelHandle, lSC));
    outputRecoEcalCandidateCollection[iSC].setSuperCluster(scRef);

    lSC++;
    iSC++;

  }

  //  Loop over Endcap SC and fill the  recoecal collection
  lSC=0; // reset local index for endcap

for(reco::SuperClusterCollection::const_iterator aClus = scEndcapHandle->begin(); aClus != scEndcapHandle->end(); aClus++) {

    const reco::Particle::Point  vtx( 0, 0, 0 );

    math::XYZVector direction =aClus->position() - vtx;
    math::XYZVector momentum = direction.unit() * aClus->energy();
    const reco::Particle::LorentzVector  p4(momentum.x(), momentum.y(), momentum.z(), aClus->energy() );

    reco::RecoEcalCandidate newCandidate(0, p4, vtx);

    outputRecoEcalCandidateCollection.push_back(newCandidate);
    reco::SuperClusterRef scRef(reco::SuperClusterRef(scEndcapHandle, lSC));
    outputRecoEcalCandidateCollection[iSC].setSuperCluster(scRef);
 
    iSC++;
    lSC++;

  }

  // put the product in the event
  outputRecoEcalCandidateCollection_p->assign(outputRecoEcalCandidateCollection.begin(),outputRecoEcalCandidateCollection.end());
  theEvent.put( outputRecoEcalCandidateCollection_p, recoEcalCandidateCollection_);

}

