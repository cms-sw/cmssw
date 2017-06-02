#ifndef PFJetsTauOverlapRemoval_H
#define PFJetsTauOverlapRemoval_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include <map>
#include <vector>
class PFJetsTauOverlapRemoval: public edm::global::EDProducer<> {
 public:
  explicit PFJetsTauOverlapRemoval(const edm::ParameterSet&);
  ~PFJetsTauOverlapRemoval();
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
    
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tauSrc;
  const edm::EDGetTokenT<reco::PFJetCollection> PFJetSrc;

};
#endif
