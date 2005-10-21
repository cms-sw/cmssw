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
// $Id: CandSelector.cc,v 1.2 2005/10/21 14:03:22 llista Exp $
//
//

#include <memory>
#include "PhysicsTools/CandAlgos/src/CandSelector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandUtils/interface/cutParser.h"
#include "PhysicsTools/CandUtils/interface/Selector.h"
#include "PhysicsTools/CandUtils/interface/candidateMethods.h"

using namespace aod;
typedef Candidate::collection Candidates;

CandSelector::CandSelector( const edm::ParameterSet& iConfig ) :
  src_(iConfig.getParameter<std::string>("src") ) {
  std::string cutString = iConfig.getParameter<std::string>("cut" );
  if( cutParser( cutString, candidateMethods(), pSelect_ ) ) {
  } else {
    throw edm::Exception(edm::errors::Configuration,"failed to parse \""+cutString+"\"");
  }
  produces<Candidates>();
}

CandSelector::~CandSelector() {
}

void
CandSelector::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;
  
  Handle<Candidates> cands;
  iEvent.getByLabel(src_,cands);
  
  std::auto_ptr<Candidates> pOut( new Candidates );
  
  for( Candidates::const_iterator c = cands->begin(); c != cands->end(); ++c ) {
    std::auto_ptr<Candidate> cand( (*c)->clone() );
    if( (*pSelect_)(*cand ) ) {
      pOut->push_back( cand.release() );
    }
  }
  
  iEvent.put( pOut );
}

