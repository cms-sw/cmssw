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
// $Id: PFCand_NoPU_WithAM.cc,v 1.1 2011/12/05 15:05:11 mgeisler Exp $
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

  typedef AssociationMap<OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;
  typedef vector<pair<TrackRef, float> > TrackQualityPairVector;

//
// constructors and destructor
//
PFCand_NoPU_WithAM::PFCand_NoPU_WithAM(const edm::ParameterSet& iConfig)
{
   //register your products

  	produces<PFCandidateCollection>();

   //now do what ever other initialization is needed

  	input_PFCandidates_ = iConfig.getParameter<InputTag>("PFCandidateCollection");
  	input_VertexCollection_ = iConfig.getParameter<InputTag>("VertexCollection");
  	input_VertexTrackAssociationMap_ = iConfig.getParameter<InputTag>("VertexTrackAssociationMap");

  	ConversionsCollection_= iConfig.getParameter<InputTag>("ConversionsCollection");

  	KshortCollection_= iConfig.getParameter<InputTag>("V0KshortCollection");
  	LambdaCollection_= iConfig.getParameter<InputTag>("V0LambdaCollection");

  	NIVertexCollection_= iConfig.getParameter<InputTag>("NIVertexCollection");
  
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
  
	//get the input pfCandidateCollection
  	Handle<PFCandidateCollection> pfCandInH;
  	iEvent.getByLabel(input_PFCandidates_,pfCandInH);
  
	//get the input vertex<->general track association map
  	Handle<TrackVertexAssMap> GTassomap;
  	iEvent.getByLabel(input_VertexTrackAssociationMap_,GTassomap);

	//get the input vertex collection
  	Handle<VertexCollection> vtxcollH;
  	iEvent.getByLabel(input_VertexCollection_,vtxcollH);

	if(vtxcollH->size()==0) return;	

	//get the conversion collection for the gamma conversions
	Handle<ConversionCollection> convCollH;
	iEvent.getByLabel(ConversionsCollection_, convCollH);

	//get the vertex composite candidate collection for the Kshort's
	Handle<VertexCompositeCandidateCollection> vertCompCandCollKshortH;
	iEvent.getByLabel(KshortCollection_, vertCompCandCollKshortH);

	//get the vertex composite candidate collection for the Lambda's
	Handle<VertexCompositeCandidateCollection> vertCompCandCollLambdaH;
	iEvent.getByLabel(LambdaCollection_, vertCompCandCollLambdaH);

	//get the displaced vertex collection for nuclear interactions
	Handle<PFDisplacedVertexCollection> displVertexCollH;
	iEvent.getByLabel(NIVertexCollection_, displVertexCollH);

	VertexRef AMfirstvertex = GTassomap->begin()->key;
	TrackQualityPairVector GTtrckcoll = GTassomap->begin()->val;
   
	for( unsigned i=0; i<pfCandInH->size(); i++ ) {
     
          PFCandidatePtr candptr(pfCandInH,i);
	  TrackRef PFCtrackref = candptr->trackRef();

	  if(PFCtrackref.isNull()){

	    VertexRef vtxref_tmp;
 	  
	    Conversion gamma;
	    VertexCompositeCandidate V0;
	    PFDisplacedVertex displVtx;
 
       	    switch( candptr->particleId() ){
       	      case PFCandidate::h:{
	        if(PFCand_NoPU_WithAM_Algos::ComesFromV0Decay(candptr,vertCompCandCollKshortH,vertCompCandCollLambdaH,vtxcollH,&vtxref_tmp)){
	          if(vtxref_tmp==AMfirstvertex) firstvertexCandidates->push_back(*candptr);
	          break;
	        }
	        if(PFCand_NoPU_WithAM_Algos::ComesFromNI(candptr,displVertexCollH,&displVtx,iSetup))
	          if(PFCand_NoPU_WithAM_Algos::FindNIVertex(candptr,displVtx,vtxcollH,true,iSetup)==AMfirstvertex) firstvertexCandidates->push_back(*candptr);
                break;
	      }
       	      case PFCandidate::gamma:{
	        if(PFCand_NoPU_WithAM_Algos::FindPFCandVertex(candptr,vtxcollH)==AMfirstvertex){
	          firstvertexCandidates->push_back(*candptr);
	        }
                break;
	      }
       	      case PFCandidate::e:{
	        if(PFCand_NoPU_WithAM_Algos::ComesFromConversion(candptr,convCollH,vtxcollH,&vtxref_tmp))
	          if(vtxref_tmp==AMfirstvertex) firstvertexCandidates->push_back(*candptr);
                break;
	      }
       	      case PFCandidate::mu:{
	        if(PFCand_NoPU_WithAM_Algos::FindPFCandVertex(candptr,vtxcollH)==AMfirstvertex) firstvertexCandidates->push_back(*candptr);
                break;
	      }
       	      case PFCandidate::h0:{
	        if(PFCand_NoPU_WithAM_Algos::ComesFromV0Decay(candptr,vertCompCandCollKshortH,vertCompCandCollLambdaH,vtxcollH,&vtxref_tmp)){
	          if(vtxref_tmp==AMfirstvertex) firstvertexCandidates->push_back(*candptr);
	          break;
	        }
	        if(PFCand_NoPU_WithAM_Algos::ComesFromNI(candptr,displVertexCollH,&displVtx,iSetup)){
	          if(PFCand_NoPU_WithAM_Algos::FindNIVertex(candptr,displVtx,vtxcollH,true,iSetup)==AMfirstvertex) firstvertexCandidates->push_back(*candptr);
	          break;
	        }
	        if(PFCand_NoPU_WithAM_Algos::FindPFCandVertex(candptr,vtxcollH)==AMfirstvertex){
	          firstvertexCandidates->push_back(*candptr);
	        }
	        break;
	      }
       	      default:
                continue;
	    }

	  }else{
  
  	    for(unsigned int index_GTtrck=0; index_GTtrck<GTtrckcoll.size(); index_GTtrck++){
 
	      TrackRef GTtrackref = GTtrckcoll.at(index_GTtrck).first;

   	      if(TrackMatch(GTtrackref,PFCtrackref)){

	        firstvertexCandidates->push_back(*candptr);
	        break;
	 	   	      
	      } 

	    }

	  }

       	} 

   	iEvent.put( firstvertexCandidates );
 
}

bool 
PFCand_NoPU_WithAM::TrackMatch(reco::TrackRef trackref1,reco::TrackRef trackref2)
{

	return (
	  (*trackref1).eta() == (*trackref2).eta() &&
	  (*trackref1).phi() == (*trackref2).phi() &&
	  (*trackref1).chi2() == (*trackref2).chi2() &&
	  (*trackref1).ndof() == (*trackref2).ndof() &&
	  (*trackref1).p() == (*trackref2).p()
	);

}

// ------------ method called once each job just before starting event loop  ------------
void 
PFCand_NoPU_WithAM::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFCand_NoPU_WithAM::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
PFCand_NoPU_WithAM::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
PFCand_NoPU_WithAM::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
PFCand_NoPU_WithAM::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
PFCand_NoPU_WithAM::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
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
