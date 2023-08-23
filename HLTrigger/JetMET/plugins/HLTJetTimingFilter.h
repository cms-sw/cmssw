/** \class HLTJetTimingFilter
 *
 *  \brief  This makes selections on the timing and associated ecal cells 
 *          produced by HLTJetTimingProducer
 *  \author Matthew Citron
 *
 */
#ifndef HLTrigger_JetMET_plugins_HLTJetTimingFilter_h
#define HLTrigger_JetMET_plugins_HLTJetTimingFilter_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template <typename T>
class HLTJetTimingFilter : public HLTFilter {
public:
  explicit HLTJetTimingFilter(const edm::ParameterSet& iConfig);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool hltFilter(edm::Event&,
                 const edm::EventSetup&,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override;

private:
  // Input collections
  const edm::InputTag jetInput_;
  const edm::EDGetTokenT<std::vector<T>> jetInputToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> jetTimesInputToken_;
  const edm::EDGetTokenT<edm::ValueMap<unsigned int>> jetCellsForTimingInputToken_;
  const edm::EDGetTokenT<edm::ValueMap<float>> jetEcalEtForTimingInputToken_;

  // Thresholds for selection
  const unsigned int minJets_;
  const double jetTimeThresh_;
  const double jetMaxTimeThresh_;
  const double jetEcalEtForTimingThresh_;
  const unsigned int jetCellsForTimingThresh_;
  const double minPt_;
};

template <typename T>
HLTJetTimingFilter<T>::HLTJetTimingFilter(const edm::ParameterSet& iConfig)
    : HLTFilter(iConfig),
      jetInput_{iConfig.getParameter<edm::InputTag>("jets")},
      jetInputToken_{consumes<std::vector<T>>(jetInput_)},
      jetTimesInputToken_{consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("jetTimes"))},
      jetCellsForTimingInputToken_{
          consumes<edm::ValueMap<unsigned int>>(iConfig.getParameter<edm::InputTag>("jetCellsForTiming"))},
      jetEcalEtForTimingInputToken_{
          consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("jetEcalEtForTiming"))},
      minJets_{iConfig.getParameter<unsigned int>("minJets")},
      jetTimeThresh_{iConfig.getParameter<double>("jetTimeThresh")},
      jetMaxTimeThresh_{iConfig.getParameter<double>("jetMaxTimeThresh")},
      jetEcalEtForTimingThresh_{iConfig.getParameter<double>("jetEcalEtForTimingThresh")},
      jetCellsForTimingThresh_{iConfig.getParameter<unsigned int>("jetCellsForTimingThresh")},
      minPt_{iConfig.getParameter<double>("minJetPt")} {}

template <typename T>
bool HLTJetTimingFilter<T>::hltFilter(edm::Event& iEvent,
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
    edm::Ref<std::vector<T>> const caloJetRef(jets, std::distance(jets->begin(), iterJet));
    if (iterJet->pt() > minPt_ and jetTimes[caloJetRef] > jetTimeThresh_ and
        jetTimes[caloJetRef] < jetMaxTimeThresh_ and jetEcalEtForTiming[caloJetRef] > jetEcalEtForTimingThresh_ and
        jetCellsForTiming[caloJetRef] > jetCellsForTimingThresh_) {
      // add caloJetRef to the event
      filterproduct.addObject(trigger::TriggerJet, caloJetRef);
      ++njets;
    }
  }

  return njets >= minJets_;
}

template <typename T>
void HLTJetTimingFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
  desc.add<double>("jetMaxTimeThresh", 999999);
  desc.add<unsigned int>("jetCellsForTimingThresh", 5);
  desc.add<double>("jetEcalEtForTimingThresh", 10.);
  desc.add<double>("minJetPt", 40.);
  descriptions.add(defaultModuleLabel<HLTJetTimingFilter<T>>(), desc);
}

#endif  // HLTrigger_JetMET_plugins_HLTJetTimingFilter_h
