#ifndef PHYSICSTOOLS_SELECTORPRODUCER_H
#define PHYSICSTOOLS_SELECTORPRODUCER_H
/*----------------------------------------------------------

$Id: SelectorProducer.h,v 1.2 2005/08/30 10:44:40 llista Exp $

  class template SelectorProducer
 
For a given selector class S, SelectorProducer<S>
is an EDProducer that takes the Candidates contained
in collection used as source, selects only the ones
that are selected by a selector instance of S, and 
stores a clone of them in a new collection.

It assumes that the selector type S implements
the follwoing interface:
  * a constructor that taks a ( const edm::ParameterSet & )
  * a bool operator()( const Candidate * c ) const

------------------------------------------------------------*/
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

template<typename S>
class SelectorProducer : public edm::EDProducer {
public:
  SelectorProducer( const edm::ParameterSet & );
  
private:
  typedef phystools::Candidate::collection Candidates;
  
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
  std::string source;
  S selector;
};

template<typename S>
SelectorProducer<S>::SelectorProducer( const edm::ParameterSet & parms ) :
  source( parms.template getParameter<std::string>( "src" ) ),
  selector( parms ) {
  produces<Candidates>();
}

template<typename S>
void SelectorProducer<S>::produce( edm::Event& evt, const edm::EventSetup& ) {
  edm::Handle<Candidates> cands;
  try {
    evt.getByLabel( source, cands );
  } catch ( std::exception e ) {
    std::cerr << "Error: can't get collection " << source << std::endl;
    return;
  }    
  
  std::auto_ptr<Candidates> selected( new Candidates );
  for( Candidates::const_iterator i = cands->begin(); i != cands->end(); ++ i ) {
    const phystools::Candidate * cand = * i;
    if ( selector( cand ) )
      selected->push_back( cand->clone() );
  }
  evt.put( selected );
}

#endif
