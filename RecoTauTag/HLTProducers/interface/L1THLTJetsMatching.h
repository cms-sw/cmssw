#ifndef L1THLTJETSMATCHING_H
#define L1THLTJETSMATCHING_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"


#include <map>
#include <vector>

class L1THLTJetsMatching: public edm::EDProducer {
 public:
  explicit L1THLTJetsMatching(const edm::ParameterSet&);
  ~L1THLTJetsMatching();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  l1t::JetVectorRef tauCandRefVec;
  l1t::JetVectorRef jetCandRefVec;
  l1t::JetVectorRef objL1CandRefVec;
  l1t::JetRef tauCandRef;

  edm::EDGetTokenT<edm::View<reco::Candidate> > jetSrc;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tauTrigger;
  double mEt_Min;
};
#endif
