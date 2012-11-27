// -*- C++ -*-
//
// Package:    PFCand_NoPU_WithAM
// Class:      PFCand_NoPU_WithAM
// 
/**\class PF_PU_AssoMap PFCand_NoPU_WithAM.cc CommonTools/RecoUtils/plugins/PFCand_NoPU_WithAM.cc

 Description: Produces a collection of PFCandidates associated to the first vertex based on the association map

*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
//         Created:  Thu Dec  1 16:07:41 CET 2011
// $Id: PFCand_NoPU_WithAM.cc,v 1.2 2012/04/18 15:11:30 mgeisler Exp $
//
//
#include "CommonTools/RecoUtils/interface/PFCand_NoPU_WithAM.h"

// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"

//
// constants, enums and typedefs
//
   
using namespace edm;
using namespace std;
using namespace reco;

  typedef AssociationMap<OneToManyWithQuality< VertexCollection, PFCandidateCollection, float> > PFCandVertexAssMap;

  typedef pair<PFCandidateRef, float> PFCandQualityPair;
  typedef vector< PFCandQualityPair > PFCandQualityPairVector;

//
// constructors and destructor
//
PFCand_NoPU_WithAM::PFCand_NoPU_WithAM(const edm::ParameterSet& iConfig)
{
   //register your products

  	produces<PFCandidateCollection>();

   //now do what ever other initialization is needed

  	input_VertexPFCandAssociationMap_ = iConfig.getParameter<InputTag>("VertexPFCandAssociationMap");
  
}


PFCand_NoPU_WithAM::~PFCand_NoPU_WithAM()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PFCand_NoPU_WithAM::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

	auto_ptr<PFCandidateCollection> firstvertexCandidates(new PFCandidateCollection() );
  
	//get the input vertex<->pfcandidate association map
  	Handle<PFCandVertexAssMap> PFCandAmH;
  	iEvent.getByLabel(input_VertexPFCandAssociationMap_,PFCandAmH);	

	PFCandQualityPairVector pfcColl; 

	if(PFCandAmH->size()!=0){
          pfcColl = PFCandAmH->begin()->val;
        }

        for(unsigned pfc_ite=0; pfc_ite<pfcColl.size(); pfc_ite++){

 	  PFCandidateRef candref = pfcColl[pfc_ite].first;  
	  firstvertexCandidates->push_back(*candref);

        }

   	iEvent.put( firstvertexCandidates );
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PFCand_NoPU_WithAM::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFCand_NoPU_WithAM);
