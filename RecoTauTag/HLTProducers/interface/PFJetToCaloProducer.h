#ifndef PFJetToCaloProducer_H
#define PFJetToCaloProducer_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"




class PFJetToCaloProducer: public edm::EDProducer {
 public:
  explicit PFJetToCaloProducer(const edm::ParameterSet&);
  ~PFJetToCaloProducer();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  edm::InputTag tauSrc_;
};
#endif
