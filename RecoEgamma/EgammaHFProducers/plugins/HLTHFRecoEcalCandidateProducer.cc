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

#include "RecoEgamma/EgammaHFProducers/plugins/HLTHFRecoEcalCandidateProducer.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

HLTHFRecoEcalCandidateProducer::HLTHFRecoEcalCandidateProducer(edm::ParameterSet const& conf):
  hfclusters_(conf.getParameter<edm::InputTag>("hfclusters")),
  HFDBversion_(conf.getUntrackedParameter<int>("HFDBversion",99)),//do nothing
  HFDBvector_(conf.getUntrackedParameter<std::vector<double> >("HFDBvector",defaultDB_)),
  hfvars_(HFDBversion_,HFDBvector_),
  algo_(true,
	conf.getParameter<double>("e9e25Cut"),
	conf.getParameter<double>("intercept2DCut"),
	conf.getParameter<double>("intercept2DSlope"),
	conf.getParameter<std::vector<double> >("e1e9Cut"),
	conf.getParameter<std::vector<double> >("eCOREe9Cut"),
	conf.getParameter<std::vector<double> >("eSeLCut"),
	hfvars_
) {

  produces<reco::RecoEcalCandidateCollection>();

} 

void HLTHFRecoEcalCandidateProducer::produce(edm::Event & e, edm::EventSetup const& iSetup) {  
  
  
  edm::Handle<reco::SuperClusterCollection> super_clus;
  edm::Handle<reco::HFEMClusterShapeAssociationCollection> hf_assoc;
 
  e.getByLabel(hfclusters_,super_clus);
  e.getByLabel(hfclusters_,hf_assoc);
 
  int nvertex = 1;
   
  // create return data
  std::auto_ptr<reco::RecoEcalCandidateCollection> retdata1(new reco::RecoEcalCandidateCollection());

  
  algo_.produce(super_clus,*hf_assoc,*retdata1,nvertex);
 
  e.put(retdata1);

}












