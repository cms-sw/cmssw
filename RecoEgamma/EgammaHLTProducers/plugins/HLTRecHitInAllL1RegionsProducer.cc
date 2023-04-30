#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"

// Reco candidates (might not need)
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

// Geometry and topology
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/Math/interface/RectangularEtaPhiRegion.h"

// Level 1 Trigger
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

//this is a re-write of HLTRechitInRegionsProducer to be able to handle arbitary L1 collections as inputs
//in the process, some of the cruft was cleaned up but it mantains almost all the old behaviour
//think the only difference now is that it wont throw if its not ECALBarrel, ECALEndcap or ECAL PS rec-hit type
class L1RegionDataBase {
public:
  virtual ~L1RegionDataBase() {}
  virtual void getEtaPhiRegions(const edm::Event&,
                                const edm::EventSetup&,
                                std::vector<RectangularEtaPhiRegion>&) const = 0;
};

template <typename T1>
class L1RegionData : public L1RegionDataBase {
private:
  double const minEt_;
  double const maxEt_;
  double const regionEtaMargin_;
  double const regionPhiMargin_;
  edm::EDGetTokenT<T1> const token_;
  edm::ESGetToken<L1CaloGeometry, L1CaloGeometryRecord> l1CaloGeometryToken_;

  void eventSetupConsumes(edm::ConsumesCollector& consumesColl);

public:
  L1RegionData(const edm::ParameterSet& para, edm::ConsumesCollector& consumesColl)
      : minEt_(para.getParameter<double>("minEt")),
        maxEt_(para.getParameter<double>("maxEt")),
        regionEtaMargin_(para.getParameter<double>("regionEtaMargin")),
        regionPhiMargin_(para.getParameter<double>("regionPhiMargin")),
        token_(consumesColl.consumes<T1>(para.getParameter<edm::InputTag>("inputColl"))) {
    eventSetupConsumes(consumesColl);
  }

  void getEtaPhiRegions(const edm::Event&,
                        const edm::EventSetup&,
                        std::vector<RectangularEtaPhiRegion>&) const override;
  template <typename T2>
  bool isEmpty(const T2& coll) const {
    return coll.empty();
  }
  template <typename T2>
  static typename T2::const_iterator beginIt(const T2& coll) {
    return coll.begin();
  }
  template <typename T2>
  static typename T2::const_iterator endIt(const T2& coll) {
    return coll.end();
  }
  template <typename T2>
  bool isEmpty(const BXVector<T2>& coll) const {
    return (coll.size() == 0 or coll.isEmpty(0));
  }
  template <typename T2>
  static typename BXVector<T2>::const_iterator beginIt(const BXVector<T2>& coll) {
    return coll.begin(0);
  }
  template <typename T2>
  static typename BXVector<T2>::const_iterator endIt(const BXVector<T2>& coll) {
    return coll.end(0);
  }
};

template <typename RecHitType>
class HLTRecHitInAllL1RegionsProducer : public edm::stream::EDProducer<> {
  using RecHitCollectionType = edm::SortedCollection<RecHitType>;

public:
  HLTRecHitInAllL1RegionsProducer(const edm::ParameterSet& ps);
  ~HLTRecHitInAllL1RegionsProducer() override {}

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  L1RegionDataBase* createL1RegionData(const std::string&,
                                       const edm::ParameterSet&,
                                       edm::ConsumesCollector&&);  //calling function owns this

  std::vector<std::unique_ptr<L1RegionDataBase>> l1RegionData_;

  std::vector<edm::InputTag> recHitLabels_;
  std::vector<std::string> productLabels_;

  std::vector<edm::EDGetTokenT<RecHitCollectionType>> recHitTokens_;

  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
};

template <typename RecHitType>
HLTRecHitInAllL1RegionsProducer<RecHitType>::HLTRecHitInAllL1RegionsProducer(const edm::ParameterSet& para)
    : caloGeometryToken_{esConsumes()} {
  const std::vector<edm::ParameterSet> l1InputRegions =
      para.getParameter<std::vector<edm::ParameterSet>>("l1InputRegions");
  for (auto& pset : l1InputRegions) {
    const std::string type = pset.getParameter<std::string>("type");
    // meh I was going to use a factory but it was going to be overly complex for my needs
    l1RegionData_.emplace_back(createL1RegionData(type, pset, consumesCollector()));
  }
  recHitLabels_ = para.getParameter<std::vector<edm::InputTag>>("recHitLabels");
  productLabels_ = para.getParameter<std::vector<std::string>>("productLabels");

  for (unsigned int collNr = 0; collNr < recHitLabels_.size(); collNr++) {
    recHitTokens_.push_back(consumes<RecHitCollectionType>(recHitLabels_[collNr]));
    produces<RecHitCollectionType>(productLabels_[collNr]);
  }
}

template <typename RecHitType>
void HLTRecHitInAllL1RegionsProducer<RecHitType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> productTags;
  productTags.push_back("EcalRegionalRecHitsEB");
  productTags.push_back("EcalRegionalRecHitsEE");
  desc.add<std::vector<std::string>>("productLabels", productTags);
  std::vector<edm::InputTag> recHitLabels;
  recHitLabels.push_back(edm::InputTag("hltEcalRegionalEgammaRecHit:EcalRecHitsEB"));
  recHitLabels.push_back(edm::InputTag("hltEcalRegionalEgammaRecHit:EcalRecHitsEE"));
  recHitLabels.push_back(edm::InputTag("hltESRegionalEgammaRecHit:EcalRecHitsES"));
  desc.add<std::vector<edm::InputTag>>("recHitLabels", recHitLabels);
  std::vector<edm::ParameterSet> l1InputRegions;

  edm::ParameterSet emIsoPSet;
  emIsoPSet.addParameter<std::string>("type", "L1EmParticle");
  emIsoPSet.addParameter<double>("minEt", 5);
  emIsoPSet.addParameter<double>("maxEt", 999);
  emIsoPSet.addParameter<double>("regionEtaMargin", 0.14);
  emIsoPSet.addParameter<double>("regionPhiMargin", 0.4);
  emIsoPSet.addParameter<edm::InputTag>("inputColl", edm::InputTag("hltL1extraParticles:NonIsolated"));
  l1InputRegions.push_back(emIsoPSet);
  edm::ParameterSet emNonIsoPSet;
  emNonIsoPSet.addParameter<std::string>("type", "L1EmParticle");
  emNonIsoPSet.addParameter<double>("minEt", 5);
  emNonIsoPSet.addParameter<double>("maxEt", 999);
  emNonIsoPSet.addParameter<double>("regionEtaMargin", 0.14);
  emNonIsoPSet.addParameter<double>("regionPhiMargin", 0.4);
  emNonIsoPSet.addParameter<edm::InputTag>("inputColl", edm::InputTag("hltL1extraParticles:Isolated"));
  l1InputRegions.push_back(emNonIsoPSet);

  // Why no Central Jets here? They are present in the python config, e.g. OnLine_HLT_GRun.py
  // SHarper: because these are the default parameters designed to reproduce the original (no jets) behaviour
  //
  edm::ParameterSet egPSet;
  egPSet.addParameter<std::string>("type", "EGamma");
  egPSet.addParameter<double>("minEt", 5);
  egPSet.addParameter<double>("maxEt", 999);
  egPSet.addParameter<double>("regionEtaMargin", 0.4);
  egPSet.addParameter<double>("regionPhiMargin", 0.5);
  egPSet.addParameter<edm::InputTag>("inputColl", edm::InputTag("hltCaloStage2Digis"));
  l1InputRegions.push_back(egPSet);

  edm::ParameterSet jetPSet;
  jetPSet.addParameter<std::string>("type", "EGamma");
  jetPSet.addParameter<double>("minEt", 200);
  jetPSet.addParameter<double>("maxEt", 999);
  jetPSet.addParameter<double>("regionEtaMargin", 0.4);
  jetPSet.addParameter<double>("regionPhiMargin", 0.5);
  jetPSet.addParameter<edm::InputTag>("inputColl", edm::InputTag("hltCaloStage2Digis"));
  l1InputRegions.push_back(jetPSet);

  edm::ParameterSetDescription l1InputRegionDesc;
  l1InputRegionDesc.add<std::string>("type");
  l1InputRegionDesc.add<double>("minEt");
  l1InputRegionDesc.add<double>("maxEt");
  l1InputRegionDesc.add<double>("regionEtaMargin");
  l1InputRegionDesc.add<double>("regionPhiMargin");
  l1InputRegionDesc.add<edm::InputTag>("inputColl");
  desc.addVPSet("l1InputRegions", l1InputRegionDesc, l1InputRegions);

  descriptions.add(defaultModuleLabel<HLTRecHitInAllL1RegionsProducer<RecHitType>>(), desc);
}

template <typename RecHitType>
void HLTRecHitInAllL1RegionsProducer<RecHitType>::produce(edm::Event& event, const edm::EventSetup& setup) {
  // get the collection geometry:
  auto const& caloGeom = setup.getData(caloGeometryToken_);

  std::vector<RectangularEtaPhiRegion> regions;
  std::for_each(l1RegionData_.begin(),
                l1RegionData_.end(),
                [&event, &setup, &regions](const std::unique_ptr<L1RegionDataBase>& input) {
                  input->getEtaPhiRegions(event, setup, regions);
                });

  for (size_t recHitCollNr = 0; recHitCollNr < recHitTokens_.size(); recHitCollNr++) {
    edm::Handle<RecHitCollectionType> recHits;
    event.getByToken(recHitTokens_[recHitCollNr], recHits);

    if (!(recHits.isValid())) {
      edm::LogError("ProductNotFound") << "could not get a handle on the " << typeid(RecHitCollectionType).name()
                                       << " named " << recHitLabels_[recHitCollNr].encode() << std::endl;
      continue;
    }

    auto filteredRecHits = std::make_unique<RecHitCollectionType>();

    if (!recHits->empty()) {
      const CaloSubdetectorGeometry* subDetGeom = caloGeom.getSubdetectorGeometry(recHits->front().id());
      if (!regions.empty()) {
        for (const RecHitType& recHit : *recHits) {
          auto this_cell = subDetGeom->getGeometry(recHit.id());
          for (const auto& region : regions) {
            if (region.inRegion(this_cell->etaPos(), this_cell->phiPos())) {
              filteredRecHits->push_back(recHit);
              break;
            }
          }
        }
      }  //end check of empty regions
    }    //end check of empty rec-hits
    //   std::cout <<"putting fileter coll in "<<filteredRecHits->size()<<std::endl;
    event.put(std::move(filteredRecHits), productLabels_[recHitCollNr]);
  }  //end loop over all rec hit collections
}

template <typename RecHitType>
L1RegionDataBase* HLTRecHitInAllL1RegionsProducer<RecHitType>::createL1RegionData(
    const std::string& type, const edm::ParameterSet& para, edm::ConsumesCollector&& consumesColl) {
  if (type == "L1EmParticle") {
    return new L1RegionData<l1extra::L1EmParticleCollection>(para, consumesColl);
  } else if (type == "L1JetParticle") {
    return new L1RegionData<l1extra::L1JetParticleCollection>(para, consumesColl);
  } else if (type == "L1MuonParticle") {
    return new L1RegionData<l1extra::L1MuonParticleCollection>(para, consumesColl);
  } else if (type == "EGamma") {
    return new L1RegionData<l1t::EGammaBxCollection>(para, consumesColl);
  } else if (type == "Jet") {
    return new L1RegionData<l1t::JetBxCollection>(para, consumesColl);
  } else if (type == "Muon") {
    return new L1RegionData<l1t::MuonBxCollection>(para, consumesColl);
  } else if (type == "Tau") {
    return new L1RegionData<l1t::TauBxCollection>(para, consumesColl);
  } else {
    //this is a major issue and could lead to rather subtle efficiency losses, so if its incorrectly configured, we're aborting the job!
    throw cms::Exception("InvalidConfig")
        << " type " << type
        << " is not recognised, this means the rec-hit you think you are keeping may not be and you should fix this "
           "error as it can lead to hard to find efficiency loses"
        << std::endl;
  }
}

template <typename L1CollType>
void L1RegionData<L1CollType>::eventSetupConsumes(edm::ConsumesCollector&) {}

template <typename L1CollType>
void L1RegionData<L1CollType>::getEtaPhiRegions(const edm::Event& event,
                                                const edm::EventSetup&,
                                                std::vector<RectangularEtaPhiRegion>& regions) const {
  edm::Handle<L1CollType> l1Cands;
  event.getByToken(token_, l1Cands);

  if (isEmpty(*l1Cands)) {
    LogDebug("HLTRecHitInAllL1RegionsProducerL1RegionData")
        << "The input collection of L1T candidates is empty (L1CollType = \""
        << edm::typeDemangle(typeid(L1CollType).name()) << "\"). No regions selected.";
    return;
  }

  for (auto l1CandIt = beginIt(*l1Cands); l1CandIt != endIt(*l1Cands); ++l1CandIt) {
    if (l1CandIt->et() >= minEt_ && l1CandIt->et() < maxEt_) {
      double etaLow = l1CandIt->eta() - regionEtaMargin_;
      double etaHigh = l1CandIt->eta() + regionEtaMargin_;
      double phiLow = l1CandIt->phi() - regionPhiMargin_;
      double phiHigh = l1CandIt->phi() + regionPhiMargin_;

      regions.push_back(RectangularEtaPhiRegion(etaLow, etaHigh, phiLow, phiHigh));
    }
  }
}

template <>
void L1RegionData<l1extra::L1JetParticleCollection>::eventSetupConsumes(edm::ConsumesCollector& consumesColl) {
  l1CaloGeometryToken_ = consumesColl.esConsumes();
}

template <>
void L1RegionData<l1extra::L1JetParticleCollection>::getEtaPhiRegions(
    const edm::Event& event, const edm::EventSetup& setup, std::vector<RectangularEtaPhiRegion>& regions) const {
  edm::Handle<l1extra::L1JetParticleCollection> l1Cands;
  event.getByToken(token_, l1Cands);

  auto const& l1CaloGeom = setup.getData(l1CaloGeometryToken_);

  for (const auto& l1Cand : *l1Cands) {
    if (l1Cand.et() >= minEt_ && l1Cand.et() < maxEt_) {
      // Access the GCT hardware object corresponding to the L1Extra EM object.
      int etaIndex = l1Cand.gctJetCand()->etaIndex();
      int phiIndex = l1Cand.gctJetCand()->phiIndex();

      // Use the L1CaloGeometry to find the eta, phi bin boundaries.
      double etaLow = l1CaloGeom.etaBinLowEdge(etaIndex);
      double etaHigh = l1CaloGeom.etaBinHighEdge(etaIndex);
      double phiLow = l1CaloGeom.emJetPhiBinLowEdge(phiIndex);
      double phiHigh = l1CaloGeom.emJetPhiBinHighEdge(phiIndex);

      etaLow -= regionEtaMargin_;
      etaHigh += regionEtaMargin_;
      phiLow -= regionPhiMargin_;
      phiHigh += regionPhiMargin_;

      regions.push_back(RectangularEtaPhiRegion(etaLow, etaHigh, phiLow, phiHigh));
    }
  }
}

template <>
void L1RegionData<l1extra::L1EmParticleCollection>::eventSetupConsumes(edm::ConsumesCollector& consumesColl) {
  l1CaloGeometryToken_ = consumesColl.esConsumes();
}

template <>
void L1RegionData<l1extra::L1EmParticleCollection>::getEtaPhiRegions(
    const edm::Event& event, const edm::EventSetup& setup, std::vector<RectangularEtaPhiRegion>& regions) const {
  edm::Handle<l1extra::L1EmParticleCollection> l1Cands;
  event.getByToken(token_, l1Cands);

  auto const& l1CaloGeom = setup.getData(l1CaloGeometryToken_);

  for (const auto& l1Cand : *l1Cands) {
    if (l1Cand.et() >= minEt_ && l1Cand.et() < maxEt_) {
      // Access the GCT hardware object corresponding to the L1Extra EM object.
      int etaIndex = l1Cand.gctEmCand()->etaIndex();
      int phiIndex = l1Cand.gctEmCand()->phiIndex();

      // Use the L1CaloGeometry to find the eta, phi bin boundaries.
      double etaLow = l1CaloGeom.etaBinLowEdge(etaIndex);
      double etaHigh = l1CaloGeom.etaBinHighEdge(etaIndex);
      double phiLow = l1CaloGeom.emJetPhiBinLowEdge(phiIndex);
      double phiHigh = l1CaloGeom.emJetPhiBinHighEdge(phiIndex);

      etaLow -= regionEtaMargin_;
      etaHigh += regionEtaMargin_;
      phiLow -= regionPhiMargin_;
      phiHigh += regionPhiMargin_;

      regions.push_back(RectangularEtaPhiRegion(etaLow, etaHigh, phiLow, phiHigh));
    }
  }
}

typedef HLTRecHitInAllL1RegionsProducer<EcalRecHit> HLTEcalRecHitInAllL1RegionsProducer;
DEFINE_FWK_MODULE(HLTEcalRecHitInAllL1RegionsProducer);

typedef HLTRecHitInAllL1RegionsProducer<EcalUncalibratedRecHit> HLTEcalUncalibratedRecHitInAllL1RegionsProducer;
DEFINE_FWK_MODULE(HLTEcalUncalibratedRecHitInAllL1RegionsProducer);
