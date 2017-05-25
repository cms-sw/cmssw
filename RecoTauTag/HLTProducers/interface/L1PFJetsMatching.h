#ifndef L1PFJetsMatching_H
#define L1PFJetsMatching_H

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
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"


#include <map>
#include <vector>
class L1PFJetsMatching: public edm::global::EDProducer<> {
 public:
  explicit L1PFJetsMatching(const edm::ParameterSet&);
  ~L1PFJetsMatching();
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
    
  const edm::EDGetTokenT<reco::PFJetCollection> jetSrc;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> jetTrigger;
  const double pt1_Min;
  const double pt2_Min;
  const double mjj_Min;
    

};
#endif
