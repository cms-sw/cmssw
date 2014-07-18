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
  HFDBversion_(conf.existsAs<bool>("HFDBversion") ? conf.getParameter<int>("HFDBversion"):99),//do nothing
  HFDBvector_(conf.existsAs<bool>("HFDBvector") ? conf.getParameter<std::vector<double> >("HFDBvector"):std::vector<double>{}),
  Cut2D_(conf.getParameter<double>("intercept2DCut")),
  defaultSlope2D_((Cut2D_<=0.83)?(0.475):((Cut2D_>0.83 && Cut2D_<=0.9)?(0.275):(0.2))),//fix for hlt unable to add slope variable now
  hfvars_(HFDBversion_,HFDBvector_),
  algo_(conf.existsAs<bool>("Correct") ? conf.getParameter<bool>("Correct") :true,
	conf.getParameter<double>("e9e25Cut"),
	conf.getParameter<double>("intercept2DCut"),
	conf.existsAs<bool>("intercept2DSlope") ? conf.getParameter<double>("intercept2DSlope") : defaultSlope2D_,
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












