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
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

HFRecoEcalCandidateProducer::HFRecoEcalCandidateProducer(edm::ParameterSet const& conf):
  hfclusters_(conf.getParameter<edm::InputTag>("hfclusters")),
  HFDBversion_(conf.getUntrackedParameter<int>("HFDBversion",99)),//do nothing
  HFDBvector_(conf.getUntrackedParameter<std::vector<double> >("HFDBvector",defaultDB_)),
  hfvars_(HFDBversion_,HFDBvector_),
  algo_(conf.getUntrackedParameter<bool>("Correct",true),
	conf.getParameter<double>("e9e25Cut"),
	conf.getParameter<double>("intercept2DCut"),
	conf.getParameter<double>("intercept2DSlope"),
	conf.getParameter<std::vector<double> >("e1e9Cut"),
	conf.getParameter<std::vector<double> >("eCOREe9Cut"),
	conf.getParameter<std::vector<double> >("eSeLCut"),
	hfvars_) 
{

  produces<reco::RecoEcalCandidateCollection>();

} 

void HFRecoEcalCandidateProducer::produce(edm::Event & e, edm::EventSetup const& iSetup) {  
  
  
  edm::Handle<reco::SuperClusterCollection> super_clus;
  edm::Handle<reco::HFEMClusterShapeAssociationCollection> hf_assoc;
 
  e.getByLabel(hfclusters_,super_clus);
  e.getByLabel(hfclusters_,hf_assoc);
 
  int nvertex = 0;
  edm:: Handle<reco::VertexCollection> pvHandle;
  e.getByLabel("offlinePrimaryVertices", pvHandle);
  const reco::VertexCollection & vertices = *pvHandle.product();
  static const int minNDOF = 4;
  static const double maxAbsZ = 15.0;
  static const double maxd0 = 2.0;
  
  //count verticies
  
  for(reco::VertexCollection::const_iterator vit = vertices.begin(); vit != vertices.end(); ++vit){
    if(vit->ndof() > minNDOF && ((maxAbsZ <= 0) || fabs(vit->z()) <= maxAbsZ) && ((maxd0 <= 0) || fabs(vit->position().rho()) <= maxd0)) 
      nvertex++;
  }
  
  // create return data
  std::auto_ptr<reco::RecoEcalCandidateCollection> retdata1(new reco::RecoEcalCandidateCollection());

  
  algo_.produce(super_clus,*hf_assoc,*retdata1,nvertex);
 
  e.put(retdata1);

}












