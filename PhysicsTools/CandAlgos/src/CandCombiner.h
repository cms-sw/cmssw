#ifndef PHYSICSTOOLS_CANDCOMBINER_H
#define PHYSICSTOOLS_CANDCOMBINER_H
// $Id: CandCombiner.h,v 1.1 2005/10/03 09:17:31 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "PhysicsTools/CandUtils/interface/TwoBodyCombiner.h"
#include "PhysicsTools/CandUtils/interface/Selector.h"
#include "PhysicsTools/CandUtils/interface/decayParser.h"
#include <string>

namespace edm {
  class ParameterSet;
}

class CandCombiner : public edm::EDProducer {
public:
  typedef aod::Candidate::collection Candidates;
  explicit CandCombiner( const edm::ParameterSet & );
  ~CandCombiner();

private:
  void produce( edm::Event& e, const edm::EventSetup& );

  std::vector<candcombiner::ConjInfo> labels_;
  std::auto_ptr<TwoBodyCombiner> combiner_;
  std::string source1_, source2_;
};

#endif
