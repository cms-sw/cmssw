// -*- C++ -*-
//
// Package:    PF_PU_AssoMap
// Class:      PF_PU_AssoMap
// 
/**\class PF_PU_AssoMap PF_PU_AssoMap.cc CommonTools/RecoUtils/plugins/PF_PU_AssoMap.cc

 Description: Produces a map with association between tracks and their particular most probable vertex with a quality of this association

*/
//
// Original Author:  Matthias Geisler,32 4-B20,+41227676487,
//

#include "CommonTools/RecoUtils/interface/PF_PU_AssoMap.h"

// system include files
#include <memory>
#include <vector>
#include <string>

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
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include <DataFormats/EgammaReco/interface/SuperCluster.h>
#include <DataFormats/EgammaReco/interface/SuperClusterFwd.h>
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertexFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include "TMath.h"
   
using namespace edm;
using namespace std;
using namespace reco;

  typedef AssociationMap<OneToManyWithQuality< VertexCollection, TrackCollection, float> > TrackVertexAssMap;

  typedef pair<TrackRef, float> TrackQualityPair;
  typedef vector<TrackQualityPair > TrackQualityPairVector;

  typedef pair<VertexRef, TrackQualityPair> VertexTrackQuality;

  typedef vector<VertexRef > VertexRefV;

//
// constructors and destructor
//
PF_PU_AssoMap::PF_PU_AssoMap(const edm::ParameterSet& iConfig)
{ 
   //now do what ever initialization is needed

  	input_VertexCollection_= iConfig.getParameter<InputTag>("VertexCollection");
  	input_TrackCollection_ = iConfig.getParameter<InputTag>("TrackCollection");

  	input_VertexAssOneDim_= iConfig.getUntrackedParameter<bool>("VertexAssOneDim", true);
  	input_VertexAssClosest_= iConfig.getUntrackedParameter<bool>("VertexAssClosest", true);
  	input_VertexAssUseAbsDistance_= iConfig.getUntrackedParameter<bool>("VertexAssUseAbsDistance", true);

  	input_GsfElectronCollection_= iConfig.getParameter<InputTag>("GsfElectronCollection");
  	ConversionsCollection_= iConfig.getParameter<InputTag>("ConversionsCollection");

  	KshortCollection_= iConfig.getParameter<InputTag>("V0KshortCollection");
  	LambdaCollection_= iConfig.getParameter<InputTag>("V0LambdaCollection");

  	NIVertexCollection_= iConfig.getParameter<InputTag>("NIVertexCollection");

	produces<TrackVertexAssMap>();
}


PF_PU_AssoMap::~PF_PU_AssoMap()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PF_PU_AssoMap::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

     	auto_ptr<TrackVertexAssMap> trackvertexass(new TrackVertexAssMap() );
    
	//get the input track collection     
  	Handle<TrackCollection> trkcollH;
  	iEvent.getByLabel(input_TrackCollection_,trkcollH);

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
 
	//get the input gsfelectron collection
  	Handle<GsfElectronCollection> gsfcollH;
  	iEvent.getByLabel(input_GsfElectronCollection_,gsfcollH);

	//get the input vertex collection
  	Handle<VertexCollection> vtxcollH;
  	iEvent.getByLabel(input_VertexCollection_,vtxcollH);

	if(vtxcollH->size()==0) return;	
	  	
	//loop over all tracks in the track collection	
  	for(unsigned int index_trck = 0;  index_trck < vtxcollH->size(); ++index_trck){

 	  TrackRef trackref = TrackRef(trkcollH,index_trck);
	  VertexTrackQuality VtxTrkQualAss;

	  //First round of association:
	  //Looking for a qualified vertex with a track weight > 0
	  VtxTrkQualAss = PF_PU_AssoMapAlgos::TrackWeightAssociation(trackref,vtxcollH);

	  if(VtxTrkQualAss.second.second<=0.00001){


	    //Test if the track comes from a gamma conversion or if it is an electron 
	    //If so reassociate the track always to the first vertex
	    Conversion gamma;
	    if((PF_PU_AssoMapAlgos::ComesFromConversion(trackref,convCollH,&gamma)) || PF_PU_AssoMapAlgos::FindRelatedElectron(trackref,gsfcollH,trkcollH)){
	      if(gamma.nTracks()==2){
    	        VtxTrkQualAss = PF_PU_AssoMapAlgos::FindConversionVertex(trackref,gamma,vtxcollH,iSetup,false);
	      }else{
  	        VtxTrkQualAss = PF_PU_AssoMapAlgos::AssociateClosest3D(trackref,vtxcollH,iSetup,false);
		if(VtxTrkQualAss.second.second==-1.) VtxTrkQualAss.second.second=-2.;
	      }

	    }

	    if(VtxTrkQualAss.second.second!=-2.){

	      //Test if the track comes from a Kshort or Lambda decay
	      //If so reassociate the track to the vertex of the V0
	      VertexCompositeCandidate V0;
	      if(PF_PU_AssoMapAlgos::ComesFromV0Decay(trackref,vertCompCandCollKshortH,vertCompCandCollLambdaH,&V0)){
	        VtxTrkQualAss = PF_PU_AssoMapAlgos::FindV0Vertex(trackref,V0,vtxcollH);	
		if(VtxTrkQualAss.second.second==-1.) VtxTrkQualAss.second.second=-2.;
	      }

	    }

	    if(VtxTrkQualAss.second.second!=-2.){

	      //Test if the track comes from a nuclear interaction
	      //If so reassociate the track to the vertex of the incomming particle
	      PFDisplacedVertex displVtx;
	      if(PF_PU_AssoMapAlgos::ComesFromNI(trackref,displVertexCollH,&displVtx)){
	        VtxTrkQualAss = PF_PU_AssoMapAlgos::FindNIVertex(trackref,displVtx,vtxcollH,true,iSetup);
 	      }
 
	    }

	    //If no vertex is found with a probability greater 0.00001
	    //and no reassociation was done look for the closest vertex in z in 3d
	    //or associate the track allways to the first vertex
	    if(VtxTrkQualAss.second.second!=-2.){

	      // if input_VertexAssOneDim_ == true association done by closest in z or allways first vertex
	      if(input_VertexAssOneDim_){ 
  	        if(input_VertexAssClosest_)
  		  //associate to closest vertex in z 
                  VtxTrkQualAss = PF_PU_AssoMapAlgos::AssociateClosestInZ(trackref,vtxcollH);
	        else{
	          //choose always the first vertex from the vertex collection & bestweight set to -1
		  VtxTrkQualAss = make_pair(VertexRef(vtxcollH,0),make_pair(trackref,-1.));
	        }
	      }
	      else
  	        VtxTrkQualAss = PF_PU_AssoMapAlgos::AssociateClosest3D(trackref,vtxcollH,iSetup,input_VertexAssUseAbsDistance_);
	    }

	  } 
	    
	  //insert the best vertex and the pair of track and the quality of this association in the map
          trackvertexass->insert(VtxTrkQualAss.first,VtxTrkQualAss.second);

	}
	  
	iEvent.put( PF_PU_AssoMapAlgos::SortAssociationMap(&(*trackvertexass)) );	

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PF_PU_AssoMap::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PF_PU_AssoMap);