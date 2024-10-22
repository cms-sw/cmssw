/** \class HLTJetTimingProducer
 *
 *  \brief  This produces timing and associated ecal cell information for calo jets 
 *  \author Matthew Citron
 *
 */
#ifndef HLTrigger_JetMET_plugins_HLTJetTimingProducer_h
#define HLTrigger_JetMET_plugins_HLTJetTimingProducer_h

#include <memory>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

template <typename T>
class HLTJetTimingProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTJetTimingProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void jetTimeFromEcalCells(const T&,
                            const edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>&,
                            const CaloGeometry&,
                            float&,
                            float&,
                            uint&);

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  // Input collections
  const edm::EDGetTokenT<std::vector<T>> jetInputToken_;
  const edm::EDGetTokenT<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>> ecalRecHitsEBToken_;
  const edm::EDGetTokenT<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>> ecalRecHitsEEToken_;

  // Include barrel, endcap jets or both
  const bool barrelJets_;
  const bool endcapJets_;

  // Configurables for timing definition
  const double ecalCellEnergyThresh_;
  const double ecalCellTimeThresh_;
  const double ecalCellTimeErrorThresh_;
  const double matchingRadius2_;
};

template <typename T>
HLTJetTimingProducer<T>::HLTJetTimingProducer(const edm::ParameterSet& iConfig)
    : caloGeometryToken_(esConsumes()),
      jetInputToken_{consumes<std::vector<T>>(iConfig.getParameter<edm::InputTag>("jets"))},
      ecalRecHitsEBToken_{consumes<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>>(
          iConfig.getParameter<edm::InputTag>("ebRecHitsColl"))},
      ecalRecHitsEEToken_{consumes<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>>(
          iConfig.getParameter<edm::InputTag>("eeRecHitsColl"))},
      barrelJets_{iConfig.getParameter<bool>("barrelJets")},
      endcapJets_{iConfig.getParameter<bool>("endcapJets")},
      ecalCellEnergyThresh_{iConfig.getParameter<double>("ecalCellEnergyThresh")},
      ecalCellTimeThresh_{iConfig.getParameter<double>("ecalCellTimeThresh")},
      ecalCellTimeErrorThresh_{iConfig.getParameter<double>("ecalCellTimeErrorThresh")},
      matchingRadius2_{std::pow(iConfig.getParameter<double>("matchingRadius"), 2)} {
  produces<edm::ValueMap<float>>("");
  produces<edm::ValueMap<unsigned int>>("jetCellsForTiming");
  produces<edm::ValueMap<float>>("jetEcalEtForTiming");
}

template <typename T>
void HLTJetTimingProducer<T>::jetTimeFromEcalCells(
    const T& jet,
    const edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>& ecalRecHits,
    const CaloGeometry& caloGeometry,
    float& weightedTimeCell,
    float& totalEmEnergyCell,
    uint& nCells) {
  for (auto const& ecalRH : ecalRecHits) {
    if (ecalRH.checkFlag(EcalRecHit::kSaturated) || ecalRH.checkFlag(EcalRecHit::kLeadingEdgeRecovered) ||
        ecalRH.checkFlag(EcalRecHit::kPoorReco) || ecalRH.checkFlag(EcalRecHit::kWeird) ||
        ecalRH.checkFlag(EcalRecHit::kDiWeird))
      continue;
    if (ecalRH.energy() < ecalCellEnergyThresh_)
      continue;
    if (ecalRH.timeError() <= 0. || ecalRH.timeError() > ecalCellTimeErrorThresh_)
      continue;
    if (std::abs(ecalRH.time()) > ecalCellTimeThresh_)
      continue;
    auto const pos = caloGeometry.getPosition(ecalRH.detid());
    if (reco::deltaR2(jet, pos) > matchingRadius2_)
      continue;
    weightedTimeCell += ecalRH.time() * ecalRH.energy() * sin(pos.theta());
    totalEmEnergyCell += ecalRH.energy() * sin(pos.theta());
    nCells++;
  }
  if (totalEmEnergyCell > 0) {
    weightedTimeCell /= totalEmEnergyCell;
  }
}

template <typename T>
void HLTJetTimingProducer<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& caloGeometry = iSetup.getData(caloGeometryToken_);
  auto const jets = iEvent.getHandle(jetInputToken_);
  auto const& ecalRecHitsEB = iEvent.get(ecalRecHitsEBToken_);
  auto const& ecalRecHitsEE = iEvent.get(ecalRecHitsEEToken_);

  std::vector<float> jetTimings;
  std::vector<unsigned int> jetCellsForTiming;
  std::vector<float> jetEcalEtForTiming;

  jetTimings.reserve(jets->size());
  jetEcalEtForTiming.reserve(jets->size());
  jetCellsForTiming.reserve(jets->size());

  for (auto const& jet : *jets) {
    float weightedTimeCell = 0;
    float totalEmEnergyCell = 0;
    unsigned int nCells = 0;
    if (barrelJets_)
      jetTimeFromEcalCells(jet, ecalRecHitsEB, caloGeometry, weightedTimeCell, totalEmEnergyCell, nCells);
    if (endcapJets_) {
      weightedTimeCell *= totalEmEnergyCell;
      jetTimeFromEcalCells(jet, ecalRecHitsEE, caloGeometry, weightedTimeCell, totalEmEnergyCell, nCells);
    }

    // If there is at least one ecal cell passing selection, calculate timing
    jetTimings.emplace_back(totalEmEnergyCell > 0 ? weightedTimeCell : -50);
    jetEcalEtForTiming.emplace_back(totalEmEnergyCell);
    jetCellsForTiming.emplace_back(nCells);
  }

  std::unique_ptr<edm::ValueMap<float>> jetTimings_out(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler jetTimings_filler(*jetTimings_out);
  jetTimings_filler.insert(jets, jetTimings.begin(), jetTimings.end());
  jetTimings_filler.fill();
  iEvent.put(std::move(jetTimings_out), "");

  std::unique_ptr<edm::ValueMap<float>> jetEcalEtForTiming_out(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler jetEcalEtForTiming_filler(*jetEcalEtForTiming_out);
  jetEcalEtForTiming_filler.insert(jets, jetEcalEtForTiming.begin(), jetEcalEtForTiming.end());
  jetEcalEtForTiming_filler.fill();
  iEvent.put(std::move(jetEcalEtForTiming_out), "jetEcalEtForTiming");

  std::unique_ptr<edm::ValueMap<unsigned int>> jetCellsForTiming_out(new edm::ValueMap<unsigned int>());
  edm::ValueMap<unsigned int>::Filler jetCellsForTiming_filler(*jetCellsForTiming_out);
  jetCellsForTiming_filler.insert(jets, jetCellsForTiming.begin(), jetCellsForTiming.end());
  jetCellsForTiming_filler.fill();
  iEvent.put(std::move(jetCellsForTiming_out), "jetCellsForTiming");
}

template <typename T>
void HLTJetTimingProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag(""));
  desc.add<bool>("barrelJets", false);
  desc.add<bool>("endcapJets", false);
  desc.add<double>("ecalCellEnergyThresh", 0.5);
  desc.add<double>("ecalCellTimeThresh", 12.5);
  desc.add<double>("ecalCellTimeErrorThresh", 100.);
  desc.add<double>("matchingRadius", 0.4);
  desc.add<edm::InputTag>("ebRecHitsColl", edm::InputTag("hltEcalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("eeRecHitsColl", edm::InputTag("hltEcalRecHit", "EcalRecHitsEE"));
  descriptions.add(defaultModuleLabel<HLTJetTimingProducer<T>>(), desc);
}

#endif  // HLTrigger_JetMET_plugins_HLTJetTimingProducer_h
