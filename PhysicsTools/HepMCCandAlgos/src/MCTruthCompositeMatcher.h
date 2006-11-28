#ifndef HepMCCandAlgos_MCTruthCompositeMatcher_h
#define HepMCCandAlgos_MCTruthCompositeMatcher_h
/* \class MCTruthCompositeMatcher
 *
 * \author Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class MCTruthCompositeMatcher : public edm::EDProducer {
public:
  explicit MCTruthCompositeMatcher( const edm::ParameterSet & );
  ~MCTruthCompositeMatcher();
private:
  edm::InputTag src_;
  edm::InputTag matchMap_;
  void produce( edm::Event & , const edm::EventSetup & );
};

#endif
