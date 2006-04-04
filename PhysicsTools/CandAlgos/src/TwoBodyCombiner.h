#ifndef CandAlgos_TwoBodyCombiner_h
#define CandAlgos_TwoBodyCombiner_h
// $Id: TwoBodyCombiner.h,v 1.6 2005/12/11 19:02:14 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
#include <string>

namespace edm {
  class ParameterSet;
}

namespace cand {
  namespace modules {
    
    class TwoBodyCombiner : public edm::EDProducer {
    public:
      explicit TwoBodyCombiner( const edm::ParameterSet & );
      ~TwoBodyCombiner();
      
    private:
      void produce( edm::Event& e, const edm::EventSetup& );
      ::TwoBodyCombiner combiner;
      std::string source1, source2;
    };

  }
}

#endif
