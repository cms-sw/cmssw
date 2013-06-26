#ifndef HLTTauProducer_H
#define HLTTauProducer_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/TauReco/interface/HLTTau.h"



class HLTTauProducer: public edm::EDProducer {
 public:
  explicit HLTTauProducer(const edm::ParameterSet&);
  ~HLTTauProducer();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  edm::InputTag emIsolatedJetsL2_;
  edm::InputTag trackIsolatedJetsL25_;
  edm::InputTag trackIsolatedJetsL3_;
  double rmin_,rmax_,matchingCone_ ,ptMinLeadTk_, signalCone_, isolationCone_, ptMin_;

};
#endif
