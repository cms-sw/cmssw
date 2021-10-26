/** \class HLTCaloJetTimingProducer
 *
 *  \brief  This produces timing and associated ecal cell information for calo jets 
 *  \author Matthew Citron
 *
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Math/interface/deltaR.h"

//
// class declaration
//
class HLTCaloJetTimingProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTCaloJetTimingProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // Input collections
  const edm::EDGetTokenT<reco::CaloJetCollection> jetInputToken_;
  const edm::EDGetTokenT<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>> ecalRecHitsEBToken_;
  const edm::EDGetTokenT<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>> ecalRecHitsEEToken_;

  // Include endcap jets or only barrel
  const bool barrelOnly_;
};

//Constructor
HLTCaloJetTimingProducer::HLTCaloJetTimingProducer(const edm::ParameterSet& iConfig) : 
  jetInputToken_{consumes<std::vector<reco::CaloJet>>(iConfig.getParameter<edm::InputTag>("jets"))},
  ecalRecHitsEBToken_{consumes<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>>(iConfig.getParameter<edm::InputTag>("ebRecHitsColl"))},
  ecalRecHitsEEToken_{consumes<edm::SortedCollection<EcalRecHit, edm::StrictWeakOrdering<EcalRecHit>>>(iConfig.getParameter<edm::InputTag>("eeRecHitsColl"))},
  barrelOnly_{iConfig.getParameter<bool>("barrelOnly")} {
  produces<edm::ValueMap<float>>("");
  produces<edm::ValueMap<unsigned int>>("jetCellsForTiming");
  produces<edm::ValueMap<float>>("jetEcalEtForTiming");
}

//Producer
void HLTCaloJetTimingProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto const& jets = iEvent.get(jetInputToken_);
  auto const& ecalRecHitsEB = iEvent.get(ecalRecHitsEBToken_);
  auto const& ecalRecHitsEE = iEvent.get(ecalRecHitsEEToken_);

  std::vector<float> jetTimings;
  std::vector<unsigned int> jetCellsForTiming;
  std::vector<float> jetEcalEtForTiming;

  jetTimings.reserve(jets.size());
  jetEcalEtForTiming.reserve(jets.size());
  jetCellsForTiming.reserve(jets.size());
  for (auto const& jet : jets) {
    float weightedTimeCell = 0;
    float totalEmEnergyCell = 0;
    unsigned int nCells = 0;
    for (auto const& ecalRH : ecalRecHitsEB) {
      if (ecalRH.checkFlag(EcalRecHit::kSaturated) || ecalRH.checkFlag(EcalRecHit::kLeadingEdgeRecovered) ||
          ecalRH.checkFlag(EcalRecHit::kPoorReco) || ecalRH.checkFlag(EcalRecHit::kWeird) ||
          ecalRH.checkFlag(EcalRecHit::kDiWeird))
        continue;
      if (ecalRH.energy() < 0.5)
        continue;
      if (ecalRH.timeError() <= 0. || ecalRH.timeError() > 100)
        continue;
      if (ecalRH.time() < -12.5 || ecalRH.time() > 12.5)
        continue;
      auto const pos = pG->getPosition(ecalRH.detid());
      if (reco::deltaR2(jet, pos) > 0.16)
        continue;
      weightedTimeCell += ecalRH.time() * ecalRH.energy() * sin(p.theta());
      totalEmEnergyCell += ecalRH.energy() * sin(p.theta());
      nCells++;
    }

    if (!barrelOnly_) {
      for (auto const& ecalRH : ecalRecHitsEE) {
        if (ecalRH.checkFlag(EcalRecHit::kSaturated) || ecalRH.checkFlag(EcalRecHit::kLeadingEdgeRecovered) ||
            ecalRH.checkFlag(EcalRecHit::kPoorReco) || ecalRH.checkFlag(EcalRecHit::kWeird) ||
            ecalRH.checkFlag(EcalRecHit::kDiWeird))
          continue;
        if (ecalRH.energy() < 0.5)
          continue;
        if (ecalRH.timeError() <= 0. || ecalRH.timeError() > 100)
          continue;
        if (ecalRH.time() < -12.5 || ecalRH.time() > 12.5)
          continue;
        auto const pos = pG->getPosition(ecalRH.detid());
        if (reco::deltaR2(jet, pos) > 0.16)
          continue;
        weightedTimeCell += ecalRH.time() * ecalRH.energy() * sin(p.theta());
        totalEmEnergyCell += ecalRH.energy() * sin(p.theta());
        nCells++;
      }
    }

    // If there is at least one ecal cell passing selection, calculate timing
    jetTimings.emplace_back(totalEmEnergyCell > 0 ? weightedTimeCell / totalEmEnergyCell : -50);
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

// Fill descriptions
void HLTCaloJetTimingProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag(""));
  desc.add<bool>("barrelOnly", false);
  desc.add<edm::InputTag>("ebRecHitsColl", edm::InputTag("hltEcalRecHit", "EcalRecHitsEB"));
  desc.add<edm::InputTag>("eeRecHitsColl", edm::InputTag("hltEcalRecHit", "EcalRecHitsEE"));
  descriptions.addWithDefaultLabel(desc);
}

// declare this class as a framework plugin
DEFINE_FWK_MODULE(HLTCaloJetTimingProducer);
