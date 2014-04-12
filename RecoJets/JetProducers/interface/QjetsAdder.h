#ifndef QjetsAdder_h
#define QjetsAdder_h

#include <memory>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "RecoJets/JetAlgorithms/interface/QjetsPlugin.h"

class QjetsAdder : public edm::EDProducer { 
public:
  explicit QjetsAdder(const edm::ParameterSet& iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    src_token_(consumes<edm::View<reco::Jet>>(src_)),
    qjetsAlgo_( iConfig.getParameter<double>("zcut"),
		iConfig.getParameter<double>("dcutfctr"),
		iConfig.getParameter<double>("expmin"),
		iConfig.getParameter<double>("expmax"),
		iConfig.getParameter<double>("rigidity")),
    ntrial_(iConfig.getParameter<int>("ntrial")),
    cutoff_(iConfig.getParameter<double>("cutoff")), 
    jetRad_(iConfig.getParameter<double>("jetRad")), 
    mJetAlgo_(iConfig.getParameter<std::string>("jetAlgo")) ,
    QjetsPreclustering_(iConfig.getParameter<int>("preclustering")) 
  {
    produces<edm::ValueMap<float> >("QjetsVolatility");
  }
  
  virtual ~QjetsAdder() {}
  
  void produce(edm::Event & iEvent, const edm::EventSetup & iSetup) ;

private:	
  edm::InputTag src_ ;
  edm::EDGetTokenT<edm::View<reco::Jet>> src_token_;
  QjetsPlugin   qjetsAlgo_ ;
  int           ntrial_;
  double        cutoff_;
  double        jetRad_;
  std::string   mJetAlgo_;
  int           QjetsPreclustering_;
  edm::Service<edm::RandomNumberGenerator> rng_;
};


#endif
