#ifndef CandAlgos_TwoBodyCombiner_h
#define CandAlgos_TwoBodyCombiner_h
// $Id: TwoBodyCombiner.h,v 1.5 2005/10/25 08:47:05 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
#include <string>

namespace edm {
  class ParameterSet;
}

namespace candmodules {

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

#endif
