#ifndef PHYSICSTOOLS_TWOBODYPRODUCER_H
#define PHYSICSTOOLS_TWOBODYPRODUCER_H
// $Id: TwoBodyProducer.h,v 1.1 2005/10/03 09:17:31 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
#include <string>

namespace edm {
  class ParameterSet;
}

class TwoBodyProducer : public edm::EDProducer {
public:
  explicit TwoBodyProducer( const edm::ParameterSet & );
  ~TwoBodyProducer();

private:
  void produce( edm::Event& e, const edm::EventSetup& );
  TwoBodyCombiner combiner;
  std::string source1, source2;
};

#endif
