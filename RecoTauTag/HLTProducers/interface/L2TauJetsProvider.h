
#ifndef L2TAUJETSPROVIDER_H
#define L2TAUJETSPROVIDER_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


class L2TauJetsProvider: public edm::EDProducer {
 public:
  explicit L2TauJetsProvider(const edm::ParameterSet&);
  ~L2TauJetsProvider();
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag jetSrc;
  edm::InputTag l1ParticleMap;
  //TO BE REPLACE WITH L1 TRIGGER BIT
  double mEt_ExtraTau;

};
#endif
