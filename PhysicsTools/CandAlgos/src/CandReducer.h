#ifndef CandAlgos_CandReducer_h
#define CandAlgos_CandReducer_h
// Ported from original implementation by Chris Jones
// $Id: CandReducer.h,v 1.4 2005/10/25 08:47:05 llista Exp $
//
#include "FWCore/Framework/interface/EDProducer.h"

namespace candmodules {

  class CandReducer : public edm::EDProducer {
  public:
    explicit CandReducer( const edm::ParameterSet& );
    ~CandReducer();
    void produce( edm::Event& evt, const edm::EventSetup& );
    std::string src_;
  };

}

#endif
