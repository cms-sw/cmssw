#ifndef RecoJets_JetProducers_plugins_FastjetJetProducer_h
#define RecoJets_JetProducers_plugins_FastjetJetProducer_h


#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"



class FastjetJetProducer : public VirtualJetProducer
{

public:
  //
  // construction/destruction
  //
  explicit FastjetJetProducer(const edm::ParameterSet& iConfig);
  virtual ~FastjetJetProducer();

  virtual void produce( edm::Event & iEvent, const edm::EventSetup & iSetup );

  
protected:

  //
  // member functions
  //

  virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup );
};


#endif
