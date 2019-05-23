/** \class HLT2jetGapFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/JetMET/interface/HLT2jetGapFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// constructors and destructor
//
HLT2jetGapFilter::HLT2jetGapFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  inputTag_ = iConfig.getParameter<edm::InputTag>("inputTag");
  minEt_ = iConfig.getParameter<double>("minEt");
  minEta_ = iConfig.getParameter<double>("minEta");

  m_theCaloJetToken = consumes<reco::CaloJetCollection>(inputTag_);
}

HLT2jetGapFilter::~HLT2jetGapFilter() = default;

void HLT2jetGapFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputTag", edm::InputTag("iterativeCone5CaloJets"));
  desc.add<double>("minEt", 90.0);
  desc.add<double>("minEta", 1.9);
  descriptions.add("hlt2jetGapFilter", desc);
}

// ------------ method called to produce the data  ------------
bool HLT2jetGapFilter::hltFilter(edm::Event& iEvent,
                                 const edm::EventSetup& iSetup,
                                 trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  using namespace trigger;

  // The filter object
  if (saveTags())
    filterproduct.addCollectionTag(inputTag_);

  edm::Handle<reco::CaloJetCollection> recocalojets;
  iEvent.getByToken(m_theCaloJetToken, recocalojets);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  if (recocalojets->size() > 1) {
    // events with two or more jets

    double etjet1 = 0.;
    double etjet2 = 0.;
    double etajet1 = 0.;
    double etajet2 = 0.;
    int countjets = 0;

    for (auto recocalojet = recocalojets->begin(); recocalojet <= (recocalojets->begin() + 1); recocalojet++) {
      if (countjets == 0) {
        etjet1 = recocalojet->et();
        etajet1 = recocalojet->eta();
      }
      if (countjets == 1) {
        etjet2 = recocalojet->et();
        etajet2 = recocalojet->eta();
      }
      countjets++;
    }

    if (etjet1 > minEt_ && etjet2 > minEt_ && (etajet1 * etajet2) < 0 && std::abs(etajet1) > minEta_ &&
        std::abs(etajet2) > minEta_) {
      for (auto recocalojet = recocalojets->begin(); recocalojet <= (recocalojets->begin() + 1); recocalojet++) {
        reco::CaloJetRef ref(reco::CaloJetRef(recocalojets, distance(recocalojets->begin(), recocalojet)));
        filterproduct.addObject(TriggerJet, ref);
        n++;
      }
    }

  }  // events with two or more jets

  // filter decision
  bool accept(n >= 2);

  return accept;
}
