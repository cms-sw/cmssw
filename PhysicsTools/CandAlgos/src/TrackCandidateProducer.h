#ifndef PHYSICSTOOLS_TRACKCANDIDATEPRODUCER_H
#define PHYSICSTOOLS_TRACKCANDIDATEPRODUCER_H
// $Id: TrackCandidateProducer.h,v 1.1 2005/07/14 11:45:30 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include <string>

namespace edm {
  class ParameterSet;
}

namespace phystools {
  
  class TrackCandidateProducer : public edm::EDProducer {
  public:
    TrackCandidateProducer( const edm::ParameterSet & );
  private:
    void produce( edm::Event& e, const edm::EventSetup& );
    std::string source;
    double massSqr;
    static const double defaultMass;
  };
}

#endif
