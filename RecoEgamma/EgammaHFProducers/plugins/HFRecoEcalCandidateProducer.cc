/** \class HFRecoEcalCandidateProducers
 *
 *  \author Kevin Klapoetke (Minnesota)
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
#include "DataFormats/EgammaReco/interface/HFEMClusterShape.h"
#include "DataFormats/EgammaReco/interface/HFEMClusterShapeAssociation.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "RecoEgamma/EgammaHFProducers/plugins/HFRecoEcalCandidateProducer.h"


HFRecoEcalCandidateProducer::HFRecoEcalCandidateProducer(edm::ParameterSet const& conf):
  hfclusters_(conf.getParameter<edm::InputTag>("hfclusters")),
  algo_(conf.getParameter<bool>("Correct"),
	conf.getParameter<double>("e9e25Cut"),
	conf.getParameter<double>("intercept2DCut"),
	conf.getParameter<std::vector<double> >("e1e9Cut"),
	conf.getParameter<std::vector<double> >("eCOREe9Cut"),
	conf.getParameter<std::vector<double> >("eSeLCut")) {

  produces<reco::RecoEcalCandidateCollection>();

} 

void HFRecoEcalCandidateProducer::produce(edm::Event & e, edm::EventSetup const& iSetup) {  
  
  
  edm::Handle<reco::SuperClusterCollection> super_clus;
  edm::Handle<reco::HFEMClusterShapeAssociationCollection> hf_assoc;
 
  e.getByLabel(hfclusters_,super_clus);
  e.getByLabel(hfclusters_,hf_assoc);

  
  
  // create return data
  std::auto_ptr<reco::RecoEcalCandidateCollection> retdata1(new reco::RecoEcalCandidateCollection());

  
  algo_.produce(super_clus,*hf_assoc,*retdata1);
 
  e.put(retdata1);

}












