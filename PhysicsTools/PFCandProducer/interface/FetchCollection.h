#ifndef _PhysicsTools_PFCandProducer_FetchCollection_
#define _PhysicsTools_PFCandProducer_FetchCollection_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

namespace pfpat {

template<class T>
void fetchCollection(T& c,
		     const edm::InputTag& tag,
		     const edm::Event& iEvent) {
  
  edm::InputTag empty;
  if( tag==empty ) return;
  
  bool found = iEvent.getByLabel(tag, c);
  
  if(!found ) {
    std::ostringstream  err;
    err<<" cannot get collection: "
       <<tag<<std::endl;
    edm::LogError("PFPAT")<<err.str();
    throw cms::Exception( "MissingProduct", err.str());
  }
  
}
 
}

#endif
