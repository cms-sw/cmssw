#ifndef PHYSICSTOOLS_TWOSAMEBODYPRODUCER_H
#define PHYSICSTOOLS_TWOSAMEBODYPRODUCER_H
// $Id: TwoSameBodyProducer.h,v 1.1 2005/07/29 07:22:52 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/CandUtils/interface/TwoSameBodyCombiner.h"
#include <string>

namespace edm {
  class ParameterSet;
}

class TwoSameBodyProducer : public edm::EDProducer {
public:
  explicit TwoSameBodyProducer( const edm::ParameterSet & );

private:
  void produce( edm::Event& e, const edm::EventSetup& );
  TwoSameBodyCombiner combiner;
  std::string source;
};

#endif
