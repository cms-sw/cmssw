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

  virtual void produceTrackJets( edm::Event & iEvent, const edm::EventSetup & iSetup );
  virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup );

 private:

  // trackjet clustering parameters
  bool useOnlyVertexTracks_;
  bool useOnlyOnePV_;
  float dzTrVtxMax_;
  float dxyTrVtxMax_;
  int minVtxNdof_;
  float maxVtxZ_;

  // jet trimming parameters
  bool useTrimming_;
  double rFilt_;
  double trimPtFracMin_;

};


#endif
