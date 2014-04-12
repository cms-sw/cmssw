#ifndef PFTauToJetProducer_H
#define PFTauToJetProducer_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TauReco/interface/PFTau.h"



class PFTauToJetProducer: public edm::EDProducer {
 public:
  explicit PFTauToJetProducer(const edm::ParameterSet&);
  ~PFTauToJetProducer();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  edm::EDGetTokenT<reco::PFTauCollection> tauSrc_;
};
#endif
