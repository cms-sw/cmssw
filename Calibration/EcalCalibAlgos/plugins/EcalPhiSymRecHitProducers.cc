// -*- C++ -*-
//
// Package:    Calibration/EcalCalibAlgos
// Class:      EcalPhiSymRecHitProducer
//
//
// Original Author:  Simone Pigazzini
//         Created:  Wed, 16 Mar 2022 15:52:48 GMT
//
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include "Calibration/EcalCalibAlgos/interface/EcalPhiSymRecHit.h"
#include "Calibration/EcalCalibAlgos/interface/EcalPhiSymInfo.h"

//---Wrapper to handle stream data
struct PhiSymCache {
  EcalPhiSymInfo ecalLumiInfo;
  EcalPhiSymRecHitCollection recHitCollEB;
  EcalPhiSymRecHitCollection recHitCollEE;

  void clear() {
    ecalLumiInfo = EcalPhiSymInfo();
    recHitCollEB.clear();
    recHitCollEE.clear();
  }
};

// cache structure for LuminosityBlock/Run Cache
struct ConfigCache {
  float etCutsEB[EcalRingCalibrationTools::N_RING_BARREL];
  float etCutsEE[EcalRingCalibrationTools::N_RING_ENDCAP];
  std::vector<DetId> barrelDetIds;
  std::vector<DetId> endcapDetIds;
};

//****************************************************************************************
// - EcalPhiSymRecHitProducerBase: base class implementing the main algorithm
// - EcalPhiSymRecHitProducerLumi: produces reduced collections per LuminosityBlock
// - EcalPhiSymRecHitProducerRun: produces reduced collections per Run
class EcalPhiSymRecHitProducerBase {
public:
  explicit EcalPhiSymRecHitProducerBase(const edm::ParameterSet& pSet, edm::ConsumesCollector&& cc);
  ~EcalPhiSymRecHitProducerBase(){};

  //---methods
  // job
  void initializeJob();
  // event
  void processEvent(edm::Event const& event,
                    edm::EventSetup const& setup,
                    ConfigCache const* config,
                    PhiSymCache* cache) const;
  // helpers
  void initializeStreamCache(ConfigCache const* config, PhiSymCache* cache) const;
  void initializePhiSymCache(edm::EventSetup const& setup,
                             edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> const& chStatusToken,
                             ConfigCache const* config,
                             std::shared_ptr<PhiSymCache>& cache) const;
  void initializeConfigCache(edm::EventSetup const& setup,
                             edm::ESGetToken<CaloGeometry, CaloGeometryRecord> const& geoToken,
                             std::shared_ptr<ConfigCache>& cache) const;
  void sumCache(PhiSymCache* summaryc, PhiSymCache* streamc) const;

  //---data memebers
  // available to derived classes
protected:
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geoToken_;
  edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> laserDbToken_;
  edm::EDGetTokenT<EBRecHitCollection> ebToken_;
  edm::EDGetTokenT<EBRecHitCollection> eeToken_;
  float etCutEB_;
  std::vector<double> eThresholdsEB_;
  float etCutEE_;
  std::vector<double> A_;
  std::vector<double> B_;
  float thrEEmod_;
  int nMisCalib_;
  int nSumEtValues_;
  std::vector<double> misCalibRangeEB_;
  std::vector<float> misCalibStepsEB_;
  std::vector<double> misCalibRangeEE_;
  std::vector<float> misCalibStepsEE_;
  //---geometry
  EcalRingCalibrationTools calibRing_;
  static const short kNRingsEB = EcalRingCalibrationTools::N_RING_BARREL;
  static const short kNRingsEE = EcalRingCalibrationTools::N_RING_ENDCAP;
  static const short ringsInOneEE = kNRingsEE / 2;
  float eThresholdsEE_[kNRingsEE];
};

//----------IMPLEMENTATION----------------------------------------------------------------
EcalPhiSymRecHitProducerBase::EcalPhiSymRecHitProducerBase(const edm::ParameterSet& pSet, edm::ConsumesCollector&& cc)
    : geoToken_(cc.esConsumes()),
      laserDbToken_(cc.esConsumes()),
      ebToken_(cc.consumes<EBRecHitCollection>(pSet.getParameter<edm::InputTag>("barrelHitCollection"))),
      eeToken_(cc.consumes<EBRecHitCollection>(pSet.getParameter<edm::InputTag>("endcapHitCollection"))),
      etCutEB_(pSet.getParameter<double>("etCut_barrel")),
      eThresholdsEB_(pSet.getParameter<std::vector<double> >("eThresholds_barrel")),
      etCutEE_(pSet.getParameter<double>("etCut_endcap")),
      A_(pSet.getParameter<std::vector<double> >("A")),
      B_(pSet.getParameter<std::vector<double> >("B")),
      thrEEmod_(pSet.getParameter<double>("thrEEmod")),
      nMisCalib_(pSet.getParameter<int>("nMisCalib") / 2),
      nSumEtValues_(nMisCalib_ * 2 + 1),
      misCalibRangeEB_(pSet.getParameter<std::vector<double> >("misCalibRangeEB")),
      misCalibRangeEE_(pSet.getParameter<std::vector<double> >("misCalibRangeEE")) {}

void EcalPhiSymRecHitProducerBase::initializeJob() {
  //---Compute the endcap thresholds using the provived parametric formula
  for (int iRing = 0; iRing < ringsInOneEE; ++iRing) {
    if (iRing < 30)
      eThresholdsEE_[iRing] = thrEEmod_ * (B_[0] + A_[0] * iRing) / 1000;
    else
      eThresholdsEE_[iRing] = thrEEmod_ * (B_[1] + A_[1] * iRing) / 1000;
    eThresholdsEE_[iRing + ringsInOneEE] = eThresholdsEE_[iRing];
  }

  //---misCalib value init (nMisCalib is half of the correct value!)
  float misCalibStepEB = std::abs(misCalibRangeEB_[1] - misCalibRangeEB_[0]) / (nMisCalib_ * 2);
  float misCalibStepEE = std::abs(misCalibRangeEE_[1] - misCalibRangeEE_[0]) / (nMisCalib_ * 2);
  misCalibStepsEB_.resize(nSumEtValues_);
  misCalibStepsEE_.resize(nSumEtValues_);
  for (int iMis = -nMisCalib_; iMis <= nMisCalib_; ++iMis) {
    //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
    int index = iMis > 0 ? iMis + nMisCalib_ : iMis == 0 ? 0 : iMis + nMisCalib_ + 1;
    misCalibStepsEB_[index] = iMis * misCalibStepEB;
    misCalibStepsEE_[index] = iMis * misCalibStepEE;
  }
}

void EcalPhiSymRecHitProducerBase::processEvent(edm::Event const& event,
                                                edm::EventSetup const& setup,
                                                ConfigCache const* configCache,
                                                PhiSymCache* streamCache) const {
  uint64_t totHitsEB = 0;
  uint64_t totHitsEE = 0;

  //---get recHits collections
  auto barrelRecHits = event.get(ebToken_);
  auto endcapRecHits = event.get(eeToken_);

  //---get the laser corrections
  edm::Timestamp evtTimeStamp(event.time().value());
  auto const& laser = setup.getData(laserDbToken_);

  //---get the geometry
  auto const& geometry = setup.getData(geoToken_);
  auto barrelGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  auto endcapGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

  //---EB---
  for (auto& recHit : barrelRecHits) {
    float energy = recHit.energy();
    EBDetId ebHit = EBDetId(recHit.id());
    int ring = calibRing_.getRingIndex(ebHit);
    //---if recHit energy is below thr even with the highest miscalib skip this recHit
    if (energy * misCalibRangeEB_[1] < eThresholdsEB_[ring])
      continue;
    float eta = barrelGeometry->getGeometry(ebHit)->getPosition().eta();

    //---compute et + miscalibration
    std::vector<float> etValues(nSumEtValues_, 0);
    //---one can do this in one for loop from -nMis to +nMis but in this way the
    //---program is faster
    //---NOTE: nMisCalib is half on the value set in the cfg python
    etValues[0] = recHit.energy() / cosh(eta);
    for (int iMis = -nMisCalib_; iMis < 0; ++iMis) {
      //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
      int index = iMis + nMisCalib_ + 1;
      etValues[index] = etValues[0] * (1 + misCalibStepsEB_[index]);
      //---set et to zero if out of range [e_thr, et_thr+1]
      if (etValues[index] * cosh(eta) < eThresholdsEB_[ring] || etValues[index] > configCache->etCutsEB[ring])
        etValues[index] = 0;
    }
    for (int iMis = 1; iMis <= nMisCalib_; ++iMis) {
      //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
      int index = iMis + nMisCalib_;
      etValues[index] = etValues[0] * (1 + misCalibStepsEB_[index]);
      //---set et to zero if out of range [e_thr, et_thr+1]
      if (etValues[index] * cosh(eta) < eThresholdsEB_[ring] || etValues[index] > configCache->etCutsEB[ring])
        etValues[index] = 0;
    }
    //---set et to zero if out of range [e_thr, et_thr+1]
    if (energy < eThresholdsEB_[ring] || etValues[0] > configCache->etCutsEB[ring])
      etValues[0] = 0;
    else
      ++totHitsEB;
    //---update the rechHit sumEt
    streamCache->recHitCollEB.at(ebHit.denseIndex())
        .addHit(etValues, laser.getLaserCorrection(recHit.id(), evtTimeStamp));
  }

  //---EE---
  for (auto& recHit : endcapRecHits) {
    EEDetId eeHit = EEDetId(recHit.id());
    int ring = calibRing_.getRingIndex(eeHit) - kNRingsEB;
    float energy = recHit.energy();
    //---if recHit energy is below thr even with the highest miscalib skip this recHit
    if (energy * misCalibRangeEE_[1] < eThresholdsEE_[ring])
      continue;
    float eta = endcapGeometry->getGeometry(eeHit)->getPosition().eta();

    //---compute et + miscalibration
    std::vector<float> etValues(nSumEtValues_, 0);
    //---one can do this in one for loop from -nMis to +nMis but in this way the
    //---program is faster
    //---NOTE: nMisCalib is half on the value set in the cfg python
    etValues[0] = recHit.energy() / cosh(eta);
    for (int iMis = -nMisCalib_; iMis < 0; ++iMis) {
      //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
      int index = iMis + nMisCalib_ + 1;
      etValues[index] = etValues[0] * (1 + misCalibStepsEE_[index]);
      //---set et to zero if out of range [e_thr, et_thr+1]
      if (etValues[index] * cosh(eta) < eThresholdsEE_[ring] || etValues[index] > configCache->etCutsEE[ring])
        etValues[index] = 0;
    }
    for (int iMis = 1; iMis <= nMisCalib_; ++iMis) {
      //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
      int index = iMis + nMisCalib_;
      etValues[index] = etValues[0] * (1 + misCalibStepsEE_[index]);
      //---set et to zero if out of range [e_thr, et_thr+1]
      if (etValues[index] * cosh(eta) < eThresholdsEE_[ring] || etValues[index] > configCache->etCutsEE[ring])
        etValues[index] = 0;
    }
    //---set et to zero if out of range [e_thr, et_thr+1]
    if (energy < eThresholdsEE_[ring] || etValues[0] > configCache->etCutsEE[ring])
      etValues[0] = 0;
    else
      ++totHitsEE;
    //---update the rechHit sumEt
    streamCache->recHitCollEE.at(eeHit.denseIndex())
        .addHit(etValues, laser.getLaserCorrection(recHit.id(), evtTimeStamp));
  }

  //---update the lumi info
  EcalPhiSymInfo thisEvent(totHitsEB, totHitsEE, 1, 0, 0, 0, 0);
  streamCache->ecalLumiInfo += thisEvent;
}

void EcalPhiSymRecHitProducerBase::initializeStreamCache(ConfigCache const* config, PhiSymCache* cache) const {
  //---Initialize the per-stream RecHitCollection
  //   both collections are initialized to contain the total
  //   number of crystals, ordered accrodingly to the hashedIndex.
  cache->clear();
  cache->recHitCollEB.resize(config->barrelDetIds.size());
  cache->recHitCollEE.resize(config->endcapDetIds.size());
  for (auto& ebDetId : config->barrelDetIds) {
    EBDetId id(ebDetId);
    cache->recHitCollEB.at(id.denseIndex()) = EcalPhiSymRecHit(id.rawId(), nSumEtValues_);
  }
  for (auto& eeDetId : config->endcapDetIds) {
    EEDetId id(eeDetId);
    cache->recHitCollEE.at(id.denseIndex()) = EcalPhiSymRecHit(id.rawId(), nSumEtValues_);
  }
}

void EcalPhiSymRecHitProducerBase::initializePhiSymCache(
    edm::EventSetup const& setup,
    edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> const& chStatusToken,
    ConfigCache const* config,
    std::shared_ptr<PhiSymCache>& cache) const {
  cache->clear();

  //---get the channels status
  auto const& chStatus = setup.getData(chStatusToken);

  cache->recHitCollEB.resize(config->barrelDetIds.size());
  cache->recHitCollEE.resize(config->endcapDetIds.size());
  for (auto& ebDetId : config->barrelDetIds) {
    EBDetId id(ebDetId);
    cache->recHitCollEB.at(id.denseIndex()) =
        EcalPhiSymRecHit(ebDetId.rawId(), nSumEtValues_, chStatus[id].getStatusCode());
  }
  for (auto& eeDetId : config->endcapDetIds) {
    EEDetId id(eeDetId);
    int ring = calibRing_.getRingIndex(id) - kNRingsEB;
    cache->recHitCollEE.at(id.denseIndex()) =
        EcalPhiSymRecHit(eeDetId.rawId(), nSumEtValues_, chStatus[id].getStatusCode());
    cache->recHitCollEE.at(id.denseIndex())
        .setEERing(ring < kNRingsEE / 2 ? ring - kNRingsEE / 2 : ring - kNRingsEE / 2 + 1);
  }
}

void EcalPhiSymRecHitProducerBase::initializeConfigCache(
    edm::EventSetup const& setup,
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> const& geoToken,
    std::shared_ptr<ConfigCache>& cache) const {
  //---get the ecal geometry
  const auto* geometry = &setup.getData(geoToken);
  calibRing_.setCaloGeometry(geometry);

  const auto* barrelGeometry = geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const auto* endcapGeometry = geometry->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  cache->barrelDetIds = barrelGeometry->getValidDetIds(DetId::Ecal, EcalBarrel);
  cache->endcapDetIds = endcapGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);

  for (auto& ebDetId : cache->barrelDetIds) {
    EBDetId id(ebDetId);
    int ring = calibRing_.getRingIndex(id);
    //---set etCut if first pass
    if (id.iphi() == 1) {
      auto cellGeometry = barrelGeometry->getGeometry(id);
      float eta = cellGeometry->getPosition().eta();
      cache->etCutsEB[ring] = eThresholdsEB_[ring] / cosh(eta) + etCutEB_;
    }
  }
  for (auto& eeDetId : cache->endcapDetIds) {
    EEDetId id(eeDetId);
    int ring = calibRing_.getRingIndex(id) - kNRingsEB;
    //---set eCutEE if first pass
    if (ring < ringsInOneEE && id.ix() == EEDetId::IX_MAX / 2) {
      auto cellGeometry = endcapGeometry->getGeometry(id);
      cache->etCutsEE[ring] = eThresholdsEE_[ring] / cosh(cellGeometry->getPosition().eta()) + etCutEE_;
      cache->etCutsEE[ring + ringsInOneEE] = cache->etCutsEE[ring];
    }
  }
}

void EcalPhiSymRecHitProducerBase::sumCache(PhiSymCache* summaryc, PhiSymCache* streamc) const {
  //---The first argument is the summary cache that
  //   contains the lumi/run summary information.
  //   The stream partial sums are passed as second argument
  summaryc->ecalLumiInfo += streamc->ecalLumiInfo;
  for (unsigned int i = 0; i < summaryc->recHitCollEB.size(); ++i)
    summaryc->recHitCollEB[i] += streamc->recHitCollEB[i];
  for (unsigned int i = 0; i < summaryc->recHitCollEE.size(); ++i)
    summaryc->recHitCollEE[i] += streamc->recHitCollEE[i];
}

//****************************************************************************************
// Lumi producer
// The StreamCache and LuminosityBlockSummaryCache contain the rec hit data, summed per
// stream, in the stream cache, and per lumi in the summary cache.
// The LuminosityBlockCache contains a set of information (detIds and thresholds)
// that requires access to the geometry record to be created. Not using the LuminosityBlockCache
// would require making the objects contained in it mutable class members which is
// discouraged.
class EcalPhiSymRecHitProducerLumi : public edm::global::EDProducer<edm::StreamCache<PhiSymCache>,
                                                                    edm::LuminosityBlockCache<ConfigCache>,
                                                                    edm::LuminosityBlockSummaryCache<PhiSymCache>,
                                                                    edm::EndLuminosityBlockProducer,
                                                                    edm::Accumulator>,
                                     public EcalPhiSymRecHitProducerBase {
public:
  explicit EcalPhiSymRecHitProducerLumi(const edm::ParameterSet& pSet);
  ~EcalPhiSymRecHitProducerLumi() override{};

private:
  //---methods
  // job
  void beginJob() override { initializeJob(); };
  // lumi
  std::shared_ptr<ConfigCache> globalBeginLuminosityBlock(edm::LuminosityBlock const& lumi,
                                                          edm::EventSetup const& setup) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) const override{};
  std::shared_ptr<PhiSymCache> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const& lumi,
                                                                 edm::EventSetup const& setup) const override;
  void globalEndLuminosityBlockSummary(edm::LuminosityBlock const& lumi,
                                       edm::EventSetup const& setup,
                                       PhiSymCache* cache) const override{};
  void globalEndLuminosityBlockProduce(edm::LuminosityBlock& lumi,
                                       edm::EventSetup const& setup,
                                       PhiSymCache const* cache) const override;
  // stream
  std::unique_ptr<PhiSymCache> beginStream(edm::StreamID stream) const override;
  void streamBeginLuminosityBlock(edm::StreamID stream,
                                  edm::LuminosityBlock const& lumi,
                                  edm::EventSetup const& setup) const override;
  void streamEndLuminosityBlockSummary(edm::StreamID stream,
                                       edm::LuminosityBlock const& lumi,
                                       edm::EventSetup const& setup,
                                       PhiSymCache* cache) const override;

  // event
  void accumulate(edm::StreamID stream, edm::Event const& event, edm::EventSetup const& setup) const override;

  // data members
  edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoTokenLumi_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> chStatusTokenLumi_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geoTokenLumi_;
};

//----------IMPLEMENTATION----------------------------------------------------------------
EcalPhiSymRecHitProducerLumi::EcalPhiSymRecHitProducerLumi(const edm::ParameterSet& pSet)
    : EcalPhiSymRecHitProducerBase(pSet, consumesCollector()),
      lhcInfoTokenLumi_(esConsumes<edm::Transition::BeginLuminosityBlock>()),
      chStatusTokenLumi_(esConsumes<edm::Transition::BeginLuminosityBlock>()),
      geoTokenLumi_(esConsumes<edm::Transition::BeginLuminosityBlock>()) {
  produces<EcalPhiSymInfo, edm::Transition::EndLuminosityBlock>();
  produces<EcalPhiSymRecHitCollection, edm::Transition::EndLuminosityBlock>("EB");
  produces<EcalPhiSymRecHitCollection, edm::Transition::EndLuminosityBlock>("EE");
}

std::shared_ptr<ConfigCache> EcalPhiSymRecHitProducerLumi::globalBeginLuminosityBlock(
    edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) const {
  auto cache = std::make_shared<ConfigCache>();

  //---Reset cache with config values
  initializeConfigCache(setup, geoTokenLumi_, cache);

  return cache;
}

std::shared_ptr<PhiSymCache> EcalPhiSymRecHitProducerLumi::globalBeginLuminosityBlockSummary(
    edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) const {
  auto cache = std::make_shared<PhiSymCache>();

  //---Get LHC info
  const auto& lhcinfo = setup.getData(lhcInfoTokenLumi_);
  EcalPhiSymInfo thisLumi(0, 0, 0, 1, lhcinfo.fillNumber(), lhcinfo.delivLumi(), lhcinfo.recLumi());

  //---Reset global cache
  initializePhiSymCache(setup, chStatusTokenLumi_, luminosityBlockCache(lumi.index()), cache);
  cache->ecalLumiInfo = thisLumi;

  return cache;
}

void EcalPhiSymRecHitProducerLumi::globalEndLuminosityBlockProduce(edm::LuminosityBlock& lumi,
                                                                   edm::EventSetup const& setup,
                                                                   PhiSymCache const* cache) const {
  //---put the collections in the LuminosityBlocks tree
  auto ecalLumiInfo = std::make_unique<EcalPhiSymInfo>(cache->ecalLumiInfo);
  ecalLumiInfo->setMiscalibInfo(
      nMisCalib_ * 2, misCalibRangeEB_[0], misCalibRangeEB_[1], misCalibRangeEE_[0], misCalibRangeEE_[1]);
  auto recHitCollEB =
      std::make_unique<EcalPhiSymRecHitCollection>(cache->recHitCollEB.begin(), cache->recHitCollEB.end());
  auto recHitCollEE =
      std::make_unique<EcalPhiSymRecHitCollection>(cache->recHitCollEE.begin(), cache->recHitCollEE.end());

  lumi.put(std::move(ecalLumiInfo));
  lumi.put(std::move(recHitCollEB), "EB");
  lumi.put(std::move(recHitCollEE), "EE");
}

std::unique_ptr<PhiSymCache> EcalPhiSymRecHitProducerLumi::beginStream(edm::StreamID stream) const {
  //---create stream cache
  return std::make_unique<PhiSymCache>();
}

void EcalPhiSymRecHitProducerLumi::streamBeginLuminosityBlock(edm::StreamID stream,
                                                              edm::LuminosityBlock const& lumi,
                                                              edm::EventSetup const& setup) const {
  //---Reset stream cache
  initializeStreamCache(luminosityBlockCache(lumi.index()), streamCache(stream));
}

void EcalPhiSymRecHitProducerLumi::streamEndLuminosityBlockSummary(edm::StreamID stream,
                                                                   edm::LuminosityBlock const& lumi,
                                                                   edm::EventSetup const& setup,
                                                                   PhiSymCache* scache) const {
  //---sum stream cache to summary cache
  sumCache(scache, streamCache(stream));
}

void EcalPhiSymRecHitProducerLumi::accumulate(edm::StreamID stream,
                                              edm::Event const& event,
                                              edm::EventSetup const& setup) const {
  processEvent(event, setup, luminosityBlockCache(event.getLuminosityBlock().index()), streamCache(stream));
}

//****************************************************************************************
// Run producer
// The StreamCache and RunSummaryCache contain the rec hit data, summed per
// stream, in the stream cache, and per run in the summary cache.
// The RunCache contains a set of information (detIds and thresholds)
// that requires access to the geometry record to be created. Not using the RunCache
// would require making the objects contained in it mutable class members which is
// discouraged.
class EcalPhiSymRecHitProducerRun : public edm::global::EDProducer<edm::StreamCache<PhiSymCache>,
                                                                   edm::RunCache<ConfigCache>,
                                                                   edm::RunSummaryCache<PhiSymCache>,
                                                                   edm::EndRunProducer,
                                                                   edm::Accumulator>,
                                    public EcalPhiSymRecHitProducerBase {
public:
  explicit EcalPhiSymRecHitProducerRun(const edm::ParameterSet& pSet);
  ~EcalPhiSymRecHitProducerRun() override{};

private:
  //---methods
  // job
  void beginJob() override { initializeJob(); };
  // run
  std::shared_ptr<ConfigCache> globalBeginRun(edm::Run const& run, edm::EventSetup const& setup) const override;
  std::shared_ptr<PhiSymCache> globalBeginRunSummary(edm::Run const& run, edm::EventSetup const& setup) const override;
  void globalEndRun(edm::Run const& run, edm::EventSetup const& setup) const override{};
  void globalEndRunSummary(edm::Run const& run, edm::EventSetup const& setup, PhiSymCache* cache) const override{};
  void globalEndRunProduce(edm::Run& run, edm::EventSetup const& setup, PhiSymCache const* cache) const override;
  // stream
  std::unique_ptr<PhiSymCache> beginStream(edm::StreamID stream) const override;
  void streamBeginLuminosityBlock(edm::StreamID stream,
                                  edm::LuminosityBlock const& lumi,
                                  edm::EventSetup const& setup) const override;
  void streamBeginRun(edm::StreamID stream, edm::Run const& run, edm::EventSetup const& setup) const override;
  void streamEndRunSummary(edm::StreamID stream,
                           edm::Run const& run,
                           edm::EventSetup const& setup,
                           PhiSymCache* cache) const override;
  // event
  void accumulate(edm::StreamID stream, edm::Event const& event, edm::EventSetup const& setup) const override;

  // data members
  edm::ESGetToken<LHCInfo, LHCInfoRcd> lhcInfoTokenLumi_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> chStatusTokenRun_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geoTokenRun_;
};

//----------IMPLEMENTATION----------------------------------------------------------------
EcalPhiSymRecHitProducerRun::EcalPhiSymRecHitProducerRun(const edm::ParameterSet& pSet)
    : EcalPhiSymRecHitProducerBase(pSet, consumesCollector()),
      lhcInfoTokenLumi_(esConsumes<edm::Transition::BeginLuminosityBlock>()),
      chStatusTokenRun_(esConsumes<edm::Transition::BeginRun>()),
      geoTokenRun_(esConsumes<edm::Transition::BeginRun>()) {
  produces<EcalPhiSymInfo, edm::Transition::EndRun>();
  produces<EcalPhiSymRecHitCollection, edm::Transition::EndRun>("EB");
  produces<EcalPhiSymRecHitCollection, edm::Transition::EndRun>("EE");
}

std::shared_ptr<ConfigCache> EcalPhiSymRecHitProducerRun::globalBeginRun(edm::Run const& run,
                                                                         edm::EventSetup const& setup) const {
  auto cache = std::make_shared<ConfigCache>();

  //---Reset cache with config values
  initializeConfigCache(setup, geoTokenRun_, cache);

  return cache;
}

std::shared_ptr<PhiSymCache> EcalPhiSymRecHitProducerRun::globalBeginRunSummary(edm::Run const& run,
                                                                                edm::EventSetup const& setup) const {
  auto cache = std::make_shared<PhiSymCache>();
  initializePhiSymCache(setup, chStatusTokenRun_, runCache(run.index()), cache);
  return cache;
}

void EcalPhiSymRecHitProducerRun::globalEndRunProduce(edm::Run& run,
                                                      edm::EventSetup const& setup,
                                                      PhiSymCache const* cache) const {
  //---put the collections in the Runs tree
  auto ecalLumiInfo = std::make_unique<EcalPhiSymInfo>(cache->ecalLumiInfo);
  ecalLumiInfo->setMiscalibInfo(
      nMisCalib_ * 2, misCalibRangeEB_[0], misCalibRangeEB_[1], misCalibRangeEE_[0], misCalibRangeEE_[1]);
  auto recHitCollEB =
      std::make_unique<EcalPhiSymRecHitCollection>(cache->recHitCollEB.begin(), cache->recHitCollEB.end());
  auto recHitCollEE =
      std::make_unique<EcalPhiSymRecHitCollection>(cache->recHitCollEE.begin(), cache->recHitCollEE.end());

  run.put(std::move(ecalLumiInfo));
  run.put(std::move(recHitCollEB), "EB");
  run.put(std::move(recHitCollEE), "EE");
}

std::unique_ptr<PhiSymCache> EcalPhiSymRecHitProducerRun::beginStream(edm::StreamID stream) const {
  //---create stream cache
  return std::make_unique<PhiSymCache>();
}

void EcalPhiSymRecHitProducerRun::streamBeginRun(edm::StreamID stream,
                                                 edm::Run const& run,
                                                 edm::EventSetup const& setup) const {
  //---Reset stream cache
  initializeStreamCache(runCache(run.index()), streamCache(stream));
}

void EcalPhiSymRecHitProducerRun::streamBeginLuminosityBlock(edm::StreamID stream,
                                                             edm::LuminosityBlock const& lumi,
                                                             edm::EventSetup const& setup) const {
  //---Get LHC info
  //   LHCInfo only returns the correct luminosity information
  //   for each lumisection, accessing LHCInfo at the beginning
  //   of each run would return only the luminosity info of the
  //   first LS.
  //   Therefore the LHCInfo is accessed only by the first stream
  //   each time a new LS is processed
  if (stream.value() == 0) {
    const auto& lhcinfo = setup.getData(lhcInfoTokenLumi_);
    EcalPhiSymInfo thisLumi(0, 0, 0, 1, lhcinfo.fillNumber(), lhcinfo.delivLumi(), lhcinfo.recLumi());

    streamCache(stream)->ecalLumiInfo += thisLumi;
  }
}

void EcalPhiSymRecHitProducerRun::streamEndRunSummary(edm::StreamID stream,
                                                      edm::Run const& run,
                                                      edm::EventSetup const& setup,
                                                      PhiSymCache* scache) const {
  //---sum stream cache to run cache
  sumCache(scache, streamCache(stream));
}

void EcalPhiSymRecHitProducerRun::accumulate(edm::StreamID stream,
                                             edm::Event const& event,
                                             edm::EventSetup const& setup) const {
  processEvent(event, setup, runCache(event.getRun().index()), streamCache(stream));
}

DEFINE_FWK_MODULE(EcalPhiSymRecHitProducerLumi);
DEFINE_FWK_MODULE(EcalPhiSymRecHitProducerRun);
