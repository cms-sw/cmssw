

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "EgammaHLTPixelMatchParamObjects.h"

namespace {
  //first 4 bits are sub detect of each hit (0=barrel, 1 = endcap)
  //next 8 bits are layer information (0=no hit, 1 = hit), first 4 are barrel, next 4 are endcap (theres an empty bit here
  //next 4 bits are nr of layers info
  int makeSeedInfo(const reco::ElectronSeed& seed) {
    int info = 0;
    for (size_t hitNr = 0; hitNr < seed.hitInfo().size(); hitNr++) {
      int subDetBit = 0x1 << hitNr;
      if (seed.subDet(hitNr) == PixelSubdetector::PixelEndcap)
        info |= subDetBit;
      int layerOffset = 3;
      if (seed.subDet(hitNr) == PixelSubdetector::PixelEndcap)
        layerOffset += 4;
      int layerBit = 0x1 << layerOffset << seed.layerOrDiskNr(hitNr);
      info |= layerBit;

      int nrLayersAlongTrajShifted = seed.nrLayersAlongTraj() << 12;
      info |= nrLayersAlongTrajShifted;
    }
    return info;
  }

}  // namespace

struct PixelData {
public:
  PixelData(std::string name,
            size_t hitNr,
            float (reco::ElectronSeed::*func)(size_t) const,
            const edm::Handle<reco::RecoEcalCandidateCollection>& candHandle)
      : name_(std::move(name)), hitNr_(hitNr), func_(func), val_(std::numeric_limits<float>::max()), valInfo_(0) {
    valMap_ = std::make_unique<reco::RecoEcalCandidateIsolationMap>(candHandle);
    valInfoMap_ = std::make_unique<reco::RecoEcalCandidateIsolationMap>(candHandle);
  }
  PixelData(PixelData&& rhs) = default;

  void resetVal() {
    val_ = std::numeric_limits<float>::max();
    valInfo_ = 0;
  }
  void fill(const reco::ElectronSeed& seed) {
    if (hitNr_ < seed.hitInfo().size()) {
      float seedVal = (seed.*func_)(hitNr_);
      if (std::abs(seedVal) < std::abs(val_)) {
        val_ = seedVal;
        valInfo_ = makeSeedInfo(seed);
      }
    }
  }
  void fill(const reco::RecoEcalCandidateRef& candRef) {
    valMap_->insert(candRef, val_);
    valInfoMap_->insert(candRef, valInfo_);
    val_ = std::numeric_limits<float>::max();
    valInfo_ = 0;
  }

  void putInto(edm::Event& event) {
    event.put(std::move(valMap_), name_ + std::to_string(hitNr_ + 1));
    event.put(std::move(valInfoMap_), name_ + std::to_string(hitNr_ + 1) + "Info");
  }

private:
  std::unique_ptr<reco::RecoEcalCandidateIsolationMap> valMap_;
  std::unique_ptr<reco::RecoEcalCandidateIsolationMap> valInfoMap_;
  std::string name_;
  size_t hitNr_;
  float (reco::ElectronSeed::*func_)(size_t) const;
  float val_;
  float valInfo_;
};

class EgammaHLTPixelMatchVarProducer : public edm::stream::EDProducer<> {
public:
  explicit EgammaHLTPixelMatchVarProducer(const edm::ParameterSet&);
  ~EgammaHLTPixelMatchVarProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;
  std::array<float, 4> calS2(const reco::ElectronSeed& seed, int charge) const;

private:
  // ----------member data ---------------------------

  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateToken_;
  const edm::EDGetTokenT<reco::ElectronSeedCollection> pixelSeedsToken_;

  egPM::Param<reco::ElectronSeed> dPhi1Para_;
  egPM::Param<reco::ElectronSeed> dPhi2Para_;
  egPM::Param<reco::ElectronSeed> dRZ2Para_;

  int productsToWrite_;
  size_t nrHits_;
};

EgammaHLTPixelMatchVarProducer::EgammaHLTPixelMatchVarProducer(const edm::ParameterSet& config)
    : recoEcalCandidateToken_(
          consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
      pixelSeedsToken_(
          consumes<reco::ElectronSeedCollection>(config.getParameter<edm::InputTag>("pixelSeedsProducer"))),
      dPhi1Para_(config.getParameter<edm::ParameterSet>("dPhi1SParams")),
      dPhi2Para_(config.getParameter<edm::ParameterSet>("dPhi2SParams")),
      dRZ2Para_(config.getParameter<edm::ParameterSet>("dRZ2SParams")),
      productsToWrite_(config.getParameter<int>("productsToWrite")),
      nrHits_(4)

{
  //register your products
  produces<reco::RecoEcalCandidateIsolationMap>("s2");
  if (productsToWrite_ >= 1) {
    produces<reco::RecoEcalCandidateIsolationMap>("dPhi1BestS2");
    produces<reco::RecoEcalCandidateIsolationMap>("dPhi2BestS2");
    produces<reco::RecoEcalCandidateIsolationMap>("dzBestS2");
  }
  if (productsToWrite_ >= 2) {
    //note for product names we start from index 1
    for (size_t hitNr = 1; hitNr <= nrHits_; hitNr++) {
      produces<reco::RecoEcalCandidateIsolationMap>("dPhi" + std::to_string(hitNr));
      produces<reco::RecoEcalCandidateIsolationMap>("dPhi" + std::to_string(hitNr) + "Info");
      produces<reco::RecoEcalCandidateIsolationMap>("dRZ" + std::to_string(hitNr));
      produces<reco::RecoEcalCandidateIsolationMap>("dRZ" + std::to_string(hitNr) + "Info");
    }
    produces<reco::RecoEcalCandidateIsolationMap>("nrClus");
    produces<reco::RecoEcalCandidateIsolationMap>("seedClusEFrac");
    produces<reco::RecoEcalCandidateIsolationMap>("phiWidth");
    produces<reco::RecoEcalCandidateIsolationMap>("etaWidth");
  }
}

EgammaHLTPixelMatchVarProducer::~EgammaHLTPixelMatchVarProducer() {}

void EgammaHLTPixelMatchVarProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<edm::InputTag>(("pixelSeedsProducer"), edm::InputTag("electronPixelSeeds"));

  edm::ParameterSetDescription varParamDesc;
  edm::ParameterSetDescription binParamDesc;

  auto binDescCases = "AbsEtaClus" >> (edm::ParameterDescription<double>("xMin", 0.0, true) and
                                       edm::ParameterDescription<double>("xMax", 3.0, true) and
                                       edm::ParameterDescription<int>("yMin", 0, true) and
                                       edm::ParameterDescription<int>("yMax", 99999, true) and
                                       edm::ParameterDescription<std::string>("funcType", "pol0", true) and
                                       edm::ParameterDescription<std::vector<double>>("funcParams", {0.}, true)) or
                      "AbsEtaClusPhi" >> (edm::ParameterDescription<double>("xMin", 0.0, true) and
                                          edm::ParameterDescription<double>("xMax", 3.0, true) and
                                          edm::ParameterDescription<int>("yMin", 0, true) and
                                          edm::ParameterDescription<int>("yMax", 99999, true) and
                                          edm::ParameterDescription<std::string>("funcType", "pol0", true) and
                                          edm::ParameterDescription<std::vector<double>>("funcParams", {0.}, true)) or
                      "AbsEtaClusEt" >> (edm::ParameterDescription<double>("xMin", 0.0, true) and
                                         edm::ParameterDescription<double>("xMax", 3.0, true) and
                                         edm::ParameterDescription<int>("yMin", 0, true) and
                                         edm::ParameterDescription<int>("yMax", 99999, true) and
                                         edm::ParameterDescription<std::string>("funcType", "pol0", true) and
                                         edm::ParameterDescription<std::vector<double>>("funcParams", {0.}, true));

  binParamDesc.ifValue(edm::ParameterDescription<std::string>("binType", "AbsEtaClus", true), std::move(binDescCases));

  varParamDesc.addVPSet("bins", binParamDesc);
  desc.add("dPhi1SParams", varParamDesc);
  desc.add("dPhi2SParams", varParamDesc);
  desc.add("dRZ2SParams", varParamDesc);
  desc.add<int>("productsToWrite", 0);
  descriptions.add(("hltEgammaHLTPixelMatchVarProducer"), desc);
}

void EgammaHLTPixelMatchVarProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
  iEvent.getByToken(recoEcalCandidateToken_, recoEcalCandHandle);

  edm::Handle<reco::ElectronSeedCollection> pixelSeedsHandle;
  iEvent.getByToken(pixelSeedsToken_, pixelSeedsHandle);

  if (!recoEcalCandHandle.isValid())
    return;
  else if (!pixelSeedsHandle.isValid()) {
    auto s2Map = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);
    for (unsigned int candNr = 0; candNr < recoEcalCandHandle->size(); candNr++) {
      reco::RecoEcalCandidateRef candRef(recoEcalCandHandle, candNr);
      s2Map->insert(candRef, 0);
    }
    iEvent.put(std::move(s2Map), "s2");
    return;
  }

  edm::ESHandle<TrackerTopology> trackerTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopoHandle);

  auto dPhi1BestS2Map = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);
  auto dPhi2BestS2Map = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);
  auto dzBestS2Map = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);
  auto s2Map = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);

  auto nrClusMap = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);
  auto seedClusEFracMap = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);
  auto phiWidthMap = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);
  auto etaWidthMap = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);

  std::vector<PixelData> pixelData;
  for (size_t hitNr = 0; hitNr < nrHits_; hitNr++) {
    pixelData.emplace_back(PixelData("dPhi", hitNr, &reco::ElectronSeed::dPhiBest, recoEcalCandHandle));
    pixelData.emplace_back(PixelData("dRZ", hitNr, &reco::ElectronSeed::dRZBest, recoEcalCandHandle));
  }

  for (unsigned int candNr = 0; candNr < recoEcalCandHandle->size(); candNr++) {
    reco::RecoEcalCandidateRef candRef(recoEcalCandHandle, candNr);
    reco::SuperClusterRef candSCRef = candRef->superCluster();

    std::array<float, 4> bestS2{{std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max(),
                                 std::numeric_limits<float>::max()}};
    for (auto& seed : *pixelSeedsHandle) {
      const edm::RefToBase<reco::CaloCluster>& pixelClusterRef = seed.caloCluster();
      reco::SuperClusterRef pixelSCRef = pixelClusterRef.castTo<reco::SuperClusterRef>();
      if (&(*candSCRef) == &(*pixelSCRef)) {
        std::array<float, 4> s2Data = calS2(seed, -1);
        std::array<float, 4> s2DataPos = calS2(seed, +1);
        if (s2Data[0] < bestS2[0])
          bestS2 = s2Data;
        if (s2DataPos[0] < bestS2[0])
          bestS2 = s2DataPos;

        if (productsToWrite_ >= 2) {
          for (auto& pixelDatum : pixelData) {
            pixelDatum.fill(seed);
          }
        }
      }
    }

    s2Map->insert(candRef, bestS2[0]);
    if (productsToWrite_ >= 1) {
      dPhi1BestS2Map->insert(candRef, bestS2[1]);
      dPhi2BestS2Map->insert(candRef, bestS2[2]);
      dzBestS2Map->insert(candRef, bestS2[3]);
    }
    if (productsToWrite_ >= 2) {
      nrClusMap->insert(candRef, candSCRef->clustersSize());
      float seedClusEFrac = candSCRef->rawEnergy() > 0 ? candSCRef->seed()->energy() / candSCRef->rawEnergy() : 0.;
      //       std::cout <<"cand "<<candSCRef->energy()<<" E Corr "<<candSCRef->correctedEnergyUncertainty()<<" "<<candSCRef->correctedEnergy()<<" width "<<candSCRef->phiWidth()<<std::endl;
      //  float seedClusEFrac = candSCRef->phiWidth();
      seedClusEFracMap->insert(candRef, seedClusEFrac);
      float phiWidth = candSCRef->phiWidth();
      float etaWidth = candSCRef->etaWidth();
      phiWidthMap->insert(candRef, phiWidth);
      etaWidthMap->insert(candRef, etaWidth);

      for (auto& pixelDatum : pixelData) {
        pixelDatum.fill(candRef);
      }
    }
  }

  iEvent.put(std::move(s2Map), "s2");
  if (productsToWrite_ >= 1) {
    iEvent.put(std::move(dPhi1BestS2Map), "dPhi1BestS2");
    iEvent.put(std::move(dPhi2BestS2Map), "dPhi2BestS2");
    iEvent.put(std::move(dzBestS2Map), "dzBestS2");
  }
  if (productsToWrite_ >= 2) {
    for (auto& pixelDatum : pixelData) {
      pixelDatum.putInto(iEvent);
    }
    iEvent.put(std::move(nrClusMap), "nrClus");
    iEvent.put(std::move(seedClusEFracMap), "seedClusEFrac");
    iEvent.put(std::move(phiWidthMap), "phiWidth");
    iEvent.put(std::move(etaWidthMap), "etaWidth");
  }
}

std::array<float, 4> EgammaHLTPixelMatchVarProducer::calS2(const reco::ElectronSeed& seed, int charge) const {
  const float dPhi1Const = dPhi1Para_(seed);
  const float dPhi2Const = dPhi2Para_(seed);
  const float dRZ2Const = dRZ2Para_(seed);

  float dPhi1 = (charge < 0 ? seed.dPhiNeg(0) : seed.dPhiPos(0)) / dPhi1Const;
  float dPhi2 = (charge < 0 ? seed.dPhiNeg(1) : seed.dPhiPos(1)) / dPhi2Const;
  float dRz2 = (charge < 0 ? seed.dRZNeg(1) : seed.dRZPos(1)) / dRZ2Const;

  float s2 = dPhi1 * dPhi1 + dPhi2 * dPhi2 + dRz2 * dRz2;
  return std::array<float, 4>{{s2, dPhi1, dPhi2, dRz2}};
}

DEFINE_FWK_MODULE(EgammaHLTPixelMatchVarProducer);
