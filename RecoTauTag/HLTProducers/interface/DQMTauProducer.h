#ifndef DQMTauProducer_H
#define DQMTauProducer_H

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



class DQMTauProducer: public edm::EDProducer {
 public:
  explicit DQMTauProducer(const edm::ParameterSet&);
  ~DQMTauProducer();
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

 private:
  edm::InputTag trackIsolatedJets_;
  double rmin_,rmax_,matchingCone_ ,ptMinLeadTk_, signalCone_, isolationCone_, ptMin_;

};

#endif
