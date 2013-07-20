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
// $Id: PF_PU_AssoMap.cc,v 1.8 2012/11/21 09:56:19 mgeisler Exp $
//
//

#include "CommonTools/RecoUtils/interface/PF_PU_AssoMap.h"

//
// static data member definitions
//

//
// constructors and destructor
//
PF_PU_AssoMap::PF_PU_AssoMap(const edm::ParameterSet& iConfig):PF_PU_AssoMapAlgos(iConfig)
{

   //now do what ever other initialization is needed

  	input_AssociationType_ = iConfig.getParameter<edm::InputTag>("AssociationType");

  	input_TrackCollection_ = iConfig.getParameter<edm::InputTag>("TrackCollection");

   //register your products

	if ( input_AssociationType_.label() == "TracksToVertex" ) {
  	  produces<TrackToVertexAssMap>();
	} else {
	  if ( input_AssociationType_.label() == "VertexToTracks" ) {
  	    produces<VertexToTrackAssMap>();
	  } else {
	    if ( input_AssociationType_.label() == "Both" ) {
  	      produces<TrackToVertexAssMap>();
  	      produces<VertexToTrackAssMap>();
	    } else {
	      std::cout << "No correct InputTag for AssociationType!" << std::endl;
	      std::cout << "Won't produce any AssociationMap!" << std::endl;
	    }
	  }
	}

  
}


PF_PU_AssoMap::~PF_PU_AssoMap()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PF_PU_AssoMap::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  using namespace std;
  using namespace reco;

    
  	//get the input track collection     
  	Handle<TrackCollection> trkcollH;
  	iEvent.getByLabel(input_TrackCollection_, trkcollH);
	
	string asstype = input_AssociationType_.label();

	PF_PU_AssoMapAlgos::GetInputCollections(iEvent,iSetup);

	if ( ( asstype == "TracksToVertex" ) || ( asstype == "Both" ) ) {
  	  auto_ptr<TrackToVertexAssMap> Track2Vertex = CreateTrackToVertexMap(trkcollH, iSetup);
  	  iEvent.put( SortAssociationMap( &(*Track2Vertex) ) );
	}
 
	if ( ( asstype == "VertexToTracks" ) || ( asstype == "Both" ) ) {
  	  auto_ptr<VertexToTrackAssMap> Vertex2Track = CreateVertexToTrackMap(trkcollH, iSetup);
  	  iEvent.put( Vertex2Track );  
	}
 
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
