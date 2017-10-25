#ifndef L1THLTTauMatching_H
#define L1THLTTauMatching_H

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


#include <map>
#include <vector>
class L1THLTTauMatching: public edm::global::EDProducer<> {
 public:
  explicit L1THLTTauMatching(const edm::ParameterSet&);
  ~L1THLTTauMatching() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
    
  const edm::EDGetTokenT<reco::PFTauCollection> jetSrc;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tauTrigger;
  const double mEt_Min;

};
#endif
