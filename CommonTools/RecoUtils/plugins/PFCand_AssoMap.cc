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
// $Id: PFCand_AssoMap.cc,v 1.2 2012/05/14 09:03:21 mgeisler Exp $
//
//
#include "CommonTools/RecoUtils/interface/PFCand_AssoMap.h"
#include "CommonTools/RecoUtils/interface/PF_PU_AssoMap.h"

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

  	ConversionsCollection_= iConfig.getParameter<InputTag>("ConversionsCollection");

  	KshortCollection_= iConfig.getParameter<InputTag>("V0KshortCollection");
  	LambdaCollection_= iConfig.getParameter<InputTag>("V0LambdaCollection");

  	NIVertexCollection_= iConfig.getParameter<InputTag>("NIVertexCollection");

   	UseBeamSpotCompatibility_= iConfig.getUntrackedParameter<bool>("UseBeamSpotCompatibility", false);
  	input_BeamSpot_= iConfig.getParameter<InputTag>("BeamSpot");

  	input_VertexAssClosest_= iConfig.getUntrackedParameter<bool>("VertexAssClosest", true);

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
  
  	//get the offfline beam spot
  	Handle<BeamSpot> beamspotH;
  	iEvent.getByLabel(input_BeamSpot_, beamspotH);
   
	for( unsigned i=0; i<pfCandInH->size(); i++ ) {
     
          PFCandidateRef candref(pfCandInH,i);

	  float weight;

	  TrackRef PFCtrackref = candref->trackRef();

          VertexPfcQuality VtxPfcQualAss;
	  VertexRef vtxref_tmp;
 	  
	  Conversion gamma;
	  VertexCompositeCandidate V0;
	  PFDisplacedVertex displVtx; 

	  if(PFCtrackref.isNull()){
     
            //the pfcand has no reference to a general track, therefore its mostly uncharged
            //it will allways be associated to the first vertex,
            //this was found out to be the best solution w.r.t. jet-pt response
            //weight set to -3.

            weight = -3.;              
	    VtxPfcQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(candref, weight));         

          }else{

	    //Round 1: Check for track weight in vertices
    	    VtxPfcQualAss = PFCand_NoPU_WithAM_Algos::TrackWeightAssociation(candref, vtxcollH);
    
            if ( VtxPfcQualAss.second.second <= 1.e-5 ) {

	      //Round 2a: Check for photon conversion
              if ( PFCtrackref->extra().isAvailable() ) {

        	if (PFCand_NoPU_WithAM_Algos::ComesFromConversion(candref,convCollH,vtxcollH,&vtxref_tmp)) {

                  if ( UseBeamSpotCompatibility_ ){

                    if (PFCand_NoPU_WithAM_Algos::CheckBeamSpotCompability(candref, beamspotH) ){
	              //associate to closest vertex in z 
	              VtxPfcQualAss = PFCand_NoPU_WithAM_Algos::AssociateClosestInZ(candref, vtxcollH);
	            } else {
	            //choose always the first vertex from the vertex collection & bestweight set to -2
	            VtxPfcQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(candref, -2.));
                    }

                  } else {
	
		    VtxPfcQualAss = make_pair(vtxref_tmp,make_pair(candref,-2.));

                  }

	          if ( VtxPfcQualAss.second.second == -1. ) VtxPfcQualAss.second.second = -2.;
                }
              }

              if ( VtxPfcQualAss.second.second != -2. ) {

	      	//Round 2b: Check for V0 decay
	        if(PFCand_NoPU_WithAM_Algos::ComesFromV0Decay(candref,vertCompCandCollKshortH,vertCompCandCollLambdaH,vtxcollH,&vtxref_tmp)){

                  if ( UseBeamSpotCompatibility_ ){

                    if (PFCand_NoPU_WithAM_Algos::CheckBeamSpotCompability(candref, beamspotH) ){
	              //associate to closest vertex in z 
	              VtxPfcQualAss = PFCand_NoPU_WithAM_Algos::AssociateClosestInZ(candref, vtxcollH);
	            } else {
	            //choose always the first vertex from the vertex collection & bestweight set to -2
	            VtxPfcQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(candref, -2.));
                    }

                  } else {
	
		    VtxPfcQualAss = make_pair(vtxref_tmp,make_pair(candref,-2.));

                  }

	          if ( VtxPfcQualAss.second.second == -1. ) VtxPfcQualAss.second.second = -2.;

	        }
	
              }
     
              if ( (VtxPfcQualAss.second.second != -2.) && displVtxColl ) {

	      	//Round 2c: Check for nuclear interaction
	        if ( PF_PU_AssoMapAlgos::ComesFromNI(PFCtrackref, displVertexCollH, &displVtx) ){

                  if ( UseBeamSpotCompatibility_ ){

                    if (PFCand_NoPU_WithAM_Algos::CheckBeamSpotCompability(candref, beamspotH) ){
	              //associate to closest vertex in z 
	              VtxPfcQualAss = PFCand_NoPU_WithAM_Algos::AssociateClosestInZ(candref, vtxcollH);
	            } else {
	            //choose always the first vertex from the vertex collection & bestweight set to -2
	            VtxPfcQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(candref, -2.));
                    }

                  } else {	          

                    vtxref_tmp = PFCand_NoPU_WithAM_Algos::FindNIVertex(candref,displVtx,vtxcollH);
                    VtxPfcQualAss = make_pair(vtxref_tmp,make_pair(candref, -2.));
 
                  }

	          if ( VtxPfcQualAss.second.second == -1. ) VtxPfcQualAss.second.second = -2.;

	        }
	
              }

              if ( VtxPfcQualAss.second.second != -2. ) {

	      	//Round 3: Associate to closest/first vertex

                if ( UseBeamSpotCompatibility_ ){

                  if (PFCand_NoPU_WithAM_Algos::CheckBeamSpotCompability(candref, beamspotH) ){
	            //associate to closest vertex in z 
	            VtxPfcQualAss = PFCand_NoPU_WithAM_Algos::AssociateClosestInZ(candref, vtxcollH);
	          } else {
	            //choose always the first vertex from the vertex collection & bestweight set to -2
	            VtxPfcQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(candref, -2.));
                  }

                } else {

  	          if ( input_VertexAssClosest_ ) {
	            //associate to closest vertex in z 
	            VtxPfcQualAss = PFCand_NoPU_WithAM_Algos::AssociateClosestInZ(candref, vtxcollH);
	          } else {
	            //choose always the first vertex from the vertex collection & bestweight set to -1
	            VtxPfcQualAss = make_pair(VertexRef(vtxcollH, 0), make_pair(candref, -1.));
	          }

                }
      
              }

            }

	  }

          pfCandAM->insert(VtxPfcQualAss.first,VtxPfcQualAss.second);

       	}

   	iEvent.put( PFCand_NoPU_WithAM_Algos::SortAssociationMap(&(*pfCandAM)) );

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
