// -*- C++ -*-
//
// Package:    PF_PU_AssoMap
// Class:      PF_PU_FirstVertexTracks
//
/**\class PF_PU_AssoMap PF_PU_FirstVertexTracks.cc CommonTools/RecoUtils/plugins/PF_PU_FirstVertexTracks.cc

  Description: Produces collection of tracks associated to the first vertex based on the pf_pu Association Map
*/
//
// Original Author:  Matthias Geisler
//         Created:  Wed Apr 18 14:48:37 CEST 2012
// $Id: PF_PU_FirstVertexTracks.cc,v 1.1 2012/11/21 09:57:30 mgeisler Exp $
//
//
#include "CommonTools/RecoUtils/interface/PF_PU_FirstVertexTracks.h"

// system include files
#include <vector>
#include <string>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//
// constants, enums and typedefs
//

using namespace edm;
using namespace std;
using namespace reco;

typedef vector<pair<TrackRef, int> > TrackQualityPairVector;


//
// static data member definitions
//

//
// constructors and destructor
//
PF_PU_FirstVertexTracks::PF_PU_FirstVertexTracks(const edm::ParameterSet& iConfig)
{
   //now do what ever other initialization is needed

  	input_AssociationType_ = iConfig.getParameter<edm::InputTag>("AssociationType");

  	token_TrackToVertexAssMap_ = mayConsume<TrackToVertexAssMap>(iConfig.getParameter<InputTag>("AssociationMap"));
  	token_VertexToTrackAssMap_ = mayConsume<VertexToTrackAssMap>(iConfig.getParameter<InputTag>("AssociationMap"));

  	token_generalTracksCollection_ = consumes<TrackCollection>(iConfig.getParameter<InputTag>("TrackCollection"));

  	token_VertexCollection_ = mayConsume<VertexCollection>(iConfig.getParameter<InputTag>("VertexCollection"));

  	input_MinQuality_ = iConfig.getParameter<int>("MinQuality");

   //register your products

	if ( input_AssociationType_.label() == "TracksToVertex" ) {
  	  produces<TrackCollection>("T2V");
	} else {
	  if ( input_AssociationType_.label() == "VertexToTracks" ) {
  	    produces<TrackCollection>("V2T");
	  } else {
	    if ( input_AssociationType_.label() == "Both" ) {
  	      produces<TrackCollection>("T2V");
  	      produces<TrackCollection>("V2T");
	    } else {
	      std::cout << "No correct InputTag for AssociationType!" << std::endl;
	      std::cout << "Won't produce any TrackCollection!" << std::endl;
	    }
	  }
	}

}


PF_PU_FirstVertexTracks::~PF_PU_FirstVertexTracks()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PF_PU_FirstVertexTracks::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

	auto_ptr<TrackCollection> t2v_firstvertextracks(new TrackCollection() );
	auto_ptr<TrackCollection> v2t_firstvertextracks(new TrackCollection() );

	bool t2vassmap = false;
	bool v2tassmap = false;

	//get the input vertex<->general track association map
  	Handle<TrackToVertexAssMap> t2vAM;
  	Handle<VertexToTrackAssMap> v2tAM;

	string asstype = input_AssociationType_.label();

	if ( ( asstype == "TracksToVertex" ) || ( asstype == "Both" ) ) {
          if ( iEvent.getByToken(token_TrackToVertexAssMap_, t2vAM ) ) {
	    t2vassmap = true;
	  }
	}

	if ( ( asstype == "VertexToTracks" ) || ( asstype == "Both" ) ) {
          if ( iEvent.getByToken(token_VertexToTrackAssMap_, v2tAM ) ) {
	    v2tassmap = true;
	  }
	}

	if ( !t2vassmap && !v2tassmap ) {
	  cout << "No input collection could be found" << endl;
	  return;
	}

	//get the input track collection
  	Handle<TrackCollection> input_trckcollH;
  	iEvent.getByToken(token_generalTracksCollection_,input_trckcollH);

	if ( t2vassmap ){

	  const TrackQualityPairVector trckcoll = t2vAM->begin()->val;

	  //get the tracks associated to the first vertex and store them in a track collection
	  for (unsigned int trckcoll_ite = 0; trckcoll_ite < trckcoll.size(); trckcoll_ite++){

	    float quality = trckcoll[trckcoll_ite].second;

	    if ( quality>=input_MinQuality_ ) {

 	      TrackRef AMtrkref = trckcoll[trckcoll_ite].first;

  	      for(unsigned int index_input_trck=0; index_input_trck<input_trckcollH->size(); index_input_trck++){

	        TrackRef input_trackref = TrackRef(input_trckcollH,index_input_trck);

   	        if( TrackMatch(*AMtrkref,*input_trackref) ){

	          t2v_firstvertextracks->push_back(*AMtrkref);
	          break;

	        }

	      }

	    }

	  }

          iEvent.put( t2v_firstvertextracks, "T2V" );

	}

	if ( v2tassmap ) {

	  //get the input vertex collection
  	  Handle<VertexCollection> input_vtxcollH;
  	  iEvent.getByToken(token_VertexCollection_,input_vtxcollH);

	  VertexRef firstVertexRef(input_vtxcollH,0);

	  VertexToTrackAssMap::const_iterator v2t_ite;

          for(v2t_ite=v2tAM->begin(); v2t_ite!=v2tAM->end(); v2t_ite++){

   	    TrackRef AMtrkref = v2t_ite->key;

  	    for(unsigned int index_input_trck=0; index_input_trck<input_trckcollH->size(); index_input_trck++){

	      TrackRef input_trackref = TrackRef(input_trckcollH,index_input_trck);

   	      if(TrackMatch(*AMtrkref,*input_trackref)){

    	        for(unsigned v_ite = 0; v_ite<(v2t_ite->val).size(); v_ite++){

      		  VertexRef vtxref = (v2t_ite->val)[v_ite].first;
      		  float quality = (v2t_ite->val)[v_ite].second;

	          if ( (vtxref==firstVertexRef) && (quality>=input_MinQuality_) ){
	            v2t_firstvertextracks->push_back(*AMtrkref);
	          }

	        }

	      }

	    }

	  }

          iEvent.put( v2t_firstvertextracks, "V2T" );

	}

}

bool
PF_PU_FirstVertexTracks::TrackMatch(const Track& track1,const Track& track2)
{

	return (
	  (track1).eta()  == (track2).eta() &&
	  (track1).phi()  == (track2).phi() &&
	  (track1).chi2() == (track2).chi2() &&
	  (track1).ndof() == (track2).ndof() &&
	  (track1).p()    == (track2).p()
	);

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PF_PU_FirstVertexTracks::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PF_PU_FirstVertexTracks);
