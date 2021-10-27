/** \class HLTCaloJetTimingFilter
 *
 *  \brief  This makes selections on the timing and associated ecal cells 
 *  produced by HLTCaloJetTimingProducer
 *  \author Matthew Citron
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//
class HLTCaloJetTimingFilter : public HLTFilter {
public:
  explicit HLTCaloJetTimingFilter(const edm::ParameterSet& iConfig);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  // Input collections
  const edm::InputTag jetInput_;
  const edm::EDGetTokenT<reco::CaloJetCollection> jetInputToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> jetTimesInputToken_;
  const edm::EDGetTokenT<edm::ValueMap<unsigned int>> jetCellsForTimingInputToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> jetEcalEtForTimingInputToken_;

  // Thresholds for selection
  const unsigned int minJets_;
  const double jetTimeThresh_;
  const double jetEcalEtForTimingThresh_;
  const unsigned int jetCellsForTimingThresh_;
  const double minPt_;
};

//Constructor
HLTCaloJetTimingFilter::HLTCaloJetTimingFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      jetInput_{iConfig.getParameter<edm::InputTag>("jets")},
      jetInputToken_{consumes<std::vector<reco::CaloJet>>(jetInput_)},
      jetTimesInputToken_{consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("jetTimes"))},
      jetCellsForTimingInputToken_{
          consumes<edm::ValueMap<unsigned int>>(iConfig.getParameter<edm::InputTag>("jetCellsForTiming"))},
      jetEcalEtForTimingInputToken_{
          consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("jetEcalEtForTiming"))},
      minJets_{iConfig.getParameter<unsigned int>("minJets")},
      jetTimeThresh_{iConfig.getParameter<double>("jetTimeThresh")},
      jetEcalEtForTimingThresh_{iConfig.getParameter<double>("jetEcalEtForTimingThresh")},
      jetCellsForTimingThresh_{iConfig.getParameter<unsigned int>("jetCellsForTimingThresh")},
      minPt_{iConfig.getParameter<double>("minJetPt")} {}

//Filter
bool HLTCaloJetTimingFilter::hltFilter(edm::Event& iEvent,
                                       const edm::EventSetup& iSetup,
                                       trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  if (saveTags())
    filterproduct.addCollectionTag(jetInput_);

  auto const jets = iEvent.getHandle(jetInputToken_);
  auto const& jetTimes = iEvent.get(jetTimesInputToken_);
  auto const& jetCellsForTiming = iEvent.get(jetCellsForTimingInputToken_);
  auto const& jetEcalEtForTiming = iEvent.get(jetEcalEtForTimingInputToken_);

  uint njets = 0;
  for (auto iterJet = jets->begin(); iterJet != jets->end(); ++iterJet) {
    reco::CaloJetRef const caloJetRef(jets, std::distance(jets->begin(), iterJet));
    if (iterJet->pt() > minPt_ and jetTimes[caloJetRef] > jetTimeThresh_ and
        jetEcalEtForTiming[caloJetRef] > jetEcalEtForTimingThresh_ and
        jetCellsForTiming[caloJetRef] > jetCellsForTimingThresh_) {
      // add caloJetRef to the event
      filterproduct.addObject(trigger::TriggerJet, caloJetRef);
      ++njets;
    }
  }
  return njets >= minJets_;
}

// Fill descriptions
void HLTCaloJetTimingFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("jets", edm::InputTag("hltDisplacedHLTCaloJetCollectionProducerMidPt"));
  desc.add<edm::InputTag>("jetTimes", edm::InputTag("hltDisplacedHLTCaloJetCollectionProducerMidPtTiming"));
  desc.add<edm::InputTag>("jetCellsForTiming",
                          edm::InputTag("hltDisplacedHLTCaloJetCollectionProducerMidPtTiming", "jetCellsForTiming"));
  desc.add<edm::InputTag>("jetEcalEtForTiming",
                          edm::InputTag("hltDisplacedHLTCaloJetCollectionProducerMidPtTiming", "jetEcalEtForTiming"));
  desc.add<unsigned int>("minJets", 1);
  desc.add<double>("jetTimeThresh", 1.);
  desc.add<unsigned int>("jetCellsForTimingThresh", 5);
  desc.add<double>("jetEcalEtForTimingThresh", 10.);
  desc.add<double>("minJetPt", 40.);
  descriptions.addWithDefaultLabel(desc);
}

// declare this class as a framework plugin
DEFINE_FWK_MODULE(HLTCaloJetTimingFilter);
