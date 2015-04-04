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
// $Id: PFCand_AssoMap.cc,v 1.5 2012/11/21 09:52:27 mgeisler Exp $
//
//
#include "CommonTools/RecoUtils/interface/PFCand_AssoMap.h"

// system include files
#include <memory>
#include <vector>
#include <string>

// user include files

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

//
// constructors and destructor
//
PFCand_AssoMap::PFCand_AssoMap(const edm::ParameterSet& iConfig):PFCand_AssoMapAlgos(iConfig, consumesCollector())
{

   //now do what ever other initialization is needed

  	input_AssociationType_ = iConfig.getParameter<edm::InputTag>("AssociationType");

  	token_PFCandidates_ = consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCandidateCollection"));

   //register your products

	if ( input_AssociationType_.label() == "PFCandsToVertex" ) {
  	  produces<PFCandToVertexAssMap>();
	} else {
	  if ( input_AssociationType_.label() == "VertexToPFCands" ) {
  	    produces<VertexToPFCandAssMap>();
	  } else {
	    if ( input_AssociationType_.label() == "Both" ) {
  	      produces<PFCandToVertexAssMap>();
  	      produces<VertexToPFCandAssMap>();
	    } else {
	      std::cout << "No correct InputTag for AssociationType!" << std::endl;
	      std::cout << "Won't produce any AssociationMap!" << std::endl;
	    }
	  }
	}

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

  using namespace edm;
  using namespace std;
  using namespace reco;

	//get the input pfCandidateCollection
  	Handle<PFCandidateCollection> pfCandH;
  	iEvent.getByToken(token_PFCandidates_,pfCandH);

	string asstype = input_AssociationType_.label();

	PFCand_AssoMapAlgos::GetInputCollections(iEvent,iSetup);

	if ( ( asstype == "PFCandsToVertex" ) || ( asstype == "Both" ) ) {
  	  auto_ptr<PFCandToVertexAssMap> PFCand2Vertex = CreatePFCandToVertexMap(pfCandH, iSetup);
	  iEvent.put( SortPFCandAssociationMap( &(*PFCand2Vertex), &iEvent.productGetter() ) );
	}

	if ( ( asstype == "VertexToPFCands" ) || ( asstype == "Both" ) ) {
  	  auto_ptr<VertexToPFCandAssMap> Vertex2PFCand = CreateVertexToPFCandMap(pfCandH, iSetup);
  	  iEvent.put( Vertex2PFCand );
	}

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
