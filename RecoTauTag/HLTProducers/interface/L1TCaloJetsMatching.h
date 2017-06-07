#ifndef L1TCaloJetsMatching_H
#define L1TCaloJetsMatching_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

#include <map>
#include <vector>
class L1TCaloJetsMatching: public edm::global::EDProducer<> {
 public:
  explicit L1TCaloJetsMatching(const edm::ParameterSet&);
  ~L1TCaloJetsMatching();
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
    
  const edm::EDGetTokenT<reco::CaloJetCollection> jetSrc_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> jetTrigger_;
  const double mPt1_Min;
  const double mPt2_Min;
  const double mMjj_Min;    

};
#endif
