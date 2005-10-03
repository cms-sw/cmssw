#ifndef PHYSICSTOOLS_TWODIFFERENTBODYPRODUCER_H
#define PHYSICSTOOLS_TWODIFFERENTBODYPRODUCER_H
// $Id: TwoDifferentBodyProducer.h,v 1.1 2005/07/29 07:22:52 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/CandUtils/interface/TwoDifferentBodyCombiner.h"
#include <string>

namespace edm {
  class ParameterSet;
}

class TwoDifferentBodyProducer : public edm::EDProducer {
public:
  explicit TwoDifferentBodyProducer( const edm::ParameterSet & );

private:
  void produce( edm::Event& e, const edm::EventSetup& );
  TwoDifferentBodyCombiner combiner;
  std::string source1, source2;
};

#endif
