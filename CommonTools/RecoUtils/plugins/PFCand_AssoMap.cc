// -*- C++ -*-
//
// Package:    PFCand_AssoMap
// Class:      PFCand_AssoMap
// 
/**\class PFCand_AssoMap PFCand_AssoMap.cc CommonTools/RecoUtils/plugins/PFCand_AssoMap.cc

  Description: Produces a map with association between pf candidates and their particular most probable vertex with a quality of this association
*/
//
// Original Author:  Matthias Geisler
//         Created:  Wed Apr 18 14:48:37 CEST 2012
// $Id: PFCand_AssoMap.cc,v 1.1 2012/04/18 15:16:18 mgeisler Exp $
//
//
#include "CommonTools/RecoUtils/interface/PFCand_AssoMap.h"

//
// constants, enums and typedefs
//
   
using namespace edm;
using namespace std;
using namespace reco;

  typedef AssociationMap<OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;
  typedef AssociationMap<OneToManyWithQuality< VertexCollection, PFCandidateCollection, float> > PFCandVertexAssMap;

  typedef vector<pair<TrackRef, float> > TrackQualityPairVector;

  typedef pair<PFCandidateRef, float> PFCandQualityPair;
  typedef vector< PFCandQualityPair > PFCandQualityPairVector;

  typedef pair<VertexRef, PFCandQualityPair> VertexPfcQuality;


//
// static data member definitions
//

//
// constructors and destructor
//
PFCand_AssoMap::PFCand_AssoMap(const edm::ParameterSet& iConfig)
  : maxNumWarnings_(3),
    numWarnings_(0)
{
   //register your products

  	produces<PFCandVertexAssMap>();

   //now do what ever other initialization is needed

  	input_PFCandidates_ = iConfig.getParameter<InputTag>("PFCandidateCollection");
  	input_VertexCollection_ = iConfig.getParameter<InputTag>("VertexCollection");
  	input_VertexTrackAssociationMap_ = iConfig.getParameter<InputTag>("VertexTrackAssociationMap");

  	ConversionsCollection_= iConfig.getParameter<InputTag>("ConversionsCollection");

  	KshortCollection_= iConfig.getParameter<InputTag>("V0KshortCollection");
  	LambdaCollection_= iConfig.getParameter<InputTag>("V0LambdaCollection");

  	NIVertexCollection_= iConfig.getParameter<InputTag>("NIVertexCollection");

        ignoremissingpfcollection_ = iConfig.getParameter<bool>("ignoreMissingCollection");

  
}


PFCand_AssoMap::~PFCand_AssoMap()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PFCand_AssoMap::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

	auto_ptr<PFCandVertexAssMap> pfCandAM(new PFCandVertexAssMap() );
  
	//get the input pfCandidateCollection
  	Handle<PFCandidateCollection> pfCandInH;
  	iEvent.getByLabel(input_PFCandidates_,pfCandInH);
  
	//get the input vertex<->general track association map
  	Handle<TrackVertexAssMap> GTassomapH;
  	iEvent.getByLabel(input_VertexTrackAssociationMap_,GTassomapH);

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
  	//create a new bool, false if no displaced vertex collection is in the event, mostly for AOD
  	bool displVtxColl = true;
  	Handle<PFDisplacedVertexCollection> displVertexCollH;
  	if(!iEvent.getByLabel(NIVertexCollection_,displVertexCollH) && ignoremissingpfcollection_){
    	  displVtxColl = false;
  	}
   
	for( unsigned i=0; i<pfCandInH->size(); i++ ) {
     
          PFCandidateRef candref(pfCandInH,i);

	  float weight;

	  TrackRef PFCtrackref = candref->trackRef();

          VertexPfcQuality VtxPfcQualAss;

	  if(PFCtrackref.isNull()){
     
            //the pfcand has no reference to a general track, therefore its mostly uncharged

	    VertexRef vtxref_tmp;

            //weight set to -3.
            weight = -3.;   
 	  
	    Conversion gamma;
	    VertexCompositeCandidate V0;
	    PFDisplacedVertex displVtx;  

            if(PFCand_NoPU_WithAM_Algos::ComesFromV0Decay(candref,vertCompCandCollKshortH,vertCompCandCollLambdaH,vtxcollH,&vtxref_tmp)){

              VtxPfcQualAss = make_pair(vtxref_tmp,make_pair(candref,weight));
              break;

            }   
	  
	    if(displVtxColl){

              if(PFCand_NoPU_WithAM_Algos::ComesFromNI(candref,displVertexCollH,&displVtx,iSetup)){

                vtxref_tmp = PFCand_NoPU_WithAM_Algos::FindNIVertex(candref,displVtx,vtxcollH,true,iSetup);
                VtxPfcQualAss = make_pair(vtxref_tmp,make_pair(candref,weight));
                break;

	      }

            }else if ( numWarnings_ < maxNumWarnings_ ){
	      edm::LogWarning("PFCand_AssoMap::produce")
	        << "No PFDisplacedVertex Collection available in input file --> skipping check for nuclear interaction !!" << std::endl;
	      ++numWarnings_;
            } 

            if(PFCand_NoPU_WithAM_Algos::ComesFromConversion(candref,convCollH,vtxcollH,&vtxref_tmp)){

              VtxPfcQualAss = make_pair(vtxref_tmp,make_pair(candref,weight));
              break;

            }

            //the uncharged particle does not come from a V0 decay, gamma conversion or a nuclear interaction
            //association to the closest vertex in z to the pfcand's vertex
            //weight is set to -4.
            weight = -4.;   
            vtxref_tmp = PFCand_NoPU_WithAM_Algos::FindPFCandVertex(candref,vtxcollH);
            VtxPfcQualAss = make_pair(vtxref_tmp,make_pair(candref,weight));

          }else{
     
            //the pfcand has a reference to a general track
            //look for association in the general track association map

            TrackVertexAssMap::const_iterator assomap_ite;

	    bool isAssoc = false;

            for(assomap_ite=GTassomapH->begin(); assomap_ite!=GTassomapH->end(); assomap_ite++){

              VertexRef vertexref = assomap_ite->key;
	      TrackQualityPairVector GTtrckcoll = assomap_ite->val;
  
  	      for(unsigned index_GTtrck=0; index_GTtrck<GTtrckcoll.size(); index_GTtrck++){
 
	        TrackRef GTtrackref = GTtrckcoll.at(index_GTtrck).first;
                weight = GTtrckcoll.at(index_GTtrck).second;

   	        if(TrackMatch(GTtrackref,PFCtrackref)){

	          VtxPfcQualAss = make_pair(vertexref,make_pair(candref,weight));
                  isAssoc = true;
                  break;
	 	   	      
	        } 

	      }

            }

            if(!isAssoc){

              //weight is set to -4.
              weight = -4.;   
              VertexRef vtxref_tmp = PFCand_NoPU_WithAM_Algos::FindPFCandVertex(candref,vtxcollH);
              VtxPfcQualAss = make_pair(vtxref_tmp,make_pair(candref,weight));

            }

	  }

          pfCandAM->insert(VtxPfcQualAss.first,VtxPfcQualAss.second);

       	}

   	iEvent.put( PFCand_NoPU_WithAM_Algos::SortAssociationMap(&(*pfCandAM)) );

}

bool 
PFCand_AssoMap::TrackMatch(reco::TrackRef trackref1,reco::TrackRef trackref2)
{

	return (
	  (*trackref1).eta() == (*trackref2).eta() &&
	  (*trackref1).phi() == (*trackref2).phi() &&
	  (*trackref1).chi2() == (*trackref2).chi2() &&
	  (*trackref1).ndof() == (*trackref2).ndof() &&
	  (*trackref1).p() == (*trackref2).p()
	);

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PFCand_AssoMap::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFCand_AssoMap);
