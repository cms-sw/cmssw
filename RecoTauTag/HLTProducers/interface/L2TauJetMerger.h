#ifndef L2TAUJETMERGER_H
#define L2TAUJETMERGER_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

class L2TauJetMerger: public edm::EDProducer {
 public:
  explicit L2TauJetMerger(const edm::ParameterSet&);
  ~L2TauJetMerger();
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
 typedef std::vector<edm::InputTag> vtag;
  vtag jetSrc;
};
#endif
