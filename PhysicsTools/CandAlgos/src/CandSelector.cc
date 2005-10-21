// -*- C++ -*-
//
// Package:    CandSelector
// Class:      CandSelector
// 
/**\class CandSelector CandSelector.cc PhysicsTools/CandSelector/src/CandSelector.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris D Jones
//         Created:  Tue Aug  9 20:26:20 EDT 2005
// $Id: CandSelector.cc,v 1.1 2005/10/21 13:56:43 llista Exp $
//
//


// system include files
#include <memory>

// user include files

#include "PhysicsTools/CandAlgos/src/CandSelector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/cutParser.h"
#include "PhysicsTools/CandUtils/interface/Selector.h"
#include "PhysicsTools/CandUtils/interface/candidateMethods.h"
//
// class decleration
//
using namespace aod;
typedef Candidate::collection Candidates;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CandSelector::CandSelector( const edm::ParameterSet& iConfig ) :
  src_(iConfig.getParameter<std::string>("src") ) {
  
  //now do what ever initialization is needed
  CandidateMethods methods = candidateMethods();
  std::string cutString = iConfig.getParameter<std::string>("cut" );
  if( cutParser( cutString,
		 methods,
		 pSelect_ )) {
  } else {
    throw edm::Exception(edm::errors::Configuration,"failed to parse \""+cutString+"\"");
  }
  produces<Candidates>();
}


CandSelector::~CandSelector()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CandSelector::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   Handle<Candidates> cands;
   iEvent.getByLabel(src_,cands);

   std::auto_ptr<Candidates> pOut( new Candidates );

   for( Candidates::const_iterator itCand = cands->begin();
	itCand != cands->end();
	++itCand ) {
     std::auto_ptr<Candidate> cand( (*itCand)->clone() );
     if( (*pSelect_)(*cand ) ) {
       pOut->push_back( cand.release() );
     }
   }

   iEvent.put( pOut );
}

