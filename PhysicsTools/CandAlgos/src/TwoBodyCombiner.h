#ifndef PHYSICSTOOLS_TWOBODYCOMBINER_H
#define PHYSICSTOOLS_TWOBODYCOMBINER_H
// $Id: TwoBodyCombiner.h,v 1.1 2005/07/14 11:49:08 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/DSTCandUtils/interface/TrackOverlap.h"

namespace edm {
  class ParameterSet;
}

namespace phystools {

  class TwoBodyCombiner : public edm::EDProducer {
  public:
    TwoBodyCombiner( const edm::ParameterSet & );
    
  protected:
    typedef Candidate::collection Candidates;
    bool select( const Candidate &, const Candidate & ) const;
    Candidate * combine( const Candidate &, const Candidate & );
    
  private:
    virtual void produce( edm::Event& e, const edm::EventSetup& ) = 0;
    
    double mass2min, mass2max;
    bool checkCharge;
    int charge;
    AddFourMomenta addp4;
    TrackOverlap overlap;
  };
}

#endif
