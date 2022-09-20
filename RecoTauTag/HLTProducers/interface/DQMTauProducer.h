#ifndef DQMTauProducer_H
#define DQMTauProducer_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"
#include "DataFormats/TauReco/interface/L2TauInfoAssociation.h"
#include "DataFormats/TauReco/interface/HLTTau.h"

class DQMTauProducer : public edm::global::EDProducer<> {
public:
  explicit DQMTauProducer(const edm::ParameterSet&);
  ~DQMTauProducer() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::EDGetTokenT<reco::IsolatedTauTagInfoCollection> trackIsolatedJets_;
  double rmin_, rmax_, matchingCone_, ptMinLeadTk_, signalCone_, isolationCone_, ptMin_;
};

#endif
