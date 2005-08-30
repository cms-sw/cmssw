#ifndef PHYSICSTOOLS_TRACKCANDIDATEPRODUCER_H
#define PHYSICSTOOLS_TRACKCANDIDATEPRODUCER_H
// $Id: TrackCandidateProducer.h,v 1.1 2005/07/29 07:22:52 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include <string>

namespace edm {
  class ParameterSet;
}

namespace phystools {
  
  class TrackCandidateProducer : public edm::EDProducer {
  public:
    TrackCandidateProducer( const edm::ParameterSet & );

  private:
    typedef Candidate::collection Candidates;

    void produce( edm::Event& e, const edm::EventSetup& );

    std::string source;
    double massSqr;
    static const double defaultMass;
  };
}

#endif
