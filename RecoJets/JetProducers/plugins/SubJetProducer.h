#ifndef RecoJets_JetProducers_CATopJetProducer2_h
#define RecoJets_JetProducers_CATopJetProducer2_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "CompoundJetProducer.h"
#include "RecoJets/JetAlgorithms/interface/SubJetAlgorithm.h"
#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"

namespace cms
{
  class SubJetProducer : public CompoundJetProducer
  {
  public:

    SubJetProducer(const edm::ParameterSet& ps);

    virtual ~SubJetProducer() {}
    
    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
    
    virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup );

  private:
    SubJetAlgorithm        alg_;         /// The algorithm to do the work

  };

}
#endif
