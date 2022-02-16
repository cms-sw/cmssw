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

#ifndef _ECAL_PHISYM_RECHIT_PRODUCERS_
#define _ECAL_PHISYM_RECHIT_PRODUCERS_

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

using namespace std;

//---Wrapper to handle cross-stream data
struct PhiSymCache {
  mutable EcalPhiSymInfo ecalLumiInfo;
  mutable EcalPhiSymRecHitCollection recHitCollEB;
  mutable EcalPhiSymRecHitCollection recHitCollEE;

  void clear() {
    ecalLumiInfo = EcalPhiSymInfo();
    recHitCollEB.clear();
    recHitCollEE.clear();
  }
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
  void processEvent(edm::StreamID stream,
                    edm::Event const& event,
                    edm::EventSetup const& setup,
                    PhiSymCache* cache) const;
  // helpers
  void initializeStreamCache(PhiSymCache* cache) const;
  void initializeGlobalCache(edm::EventSetup const& setup,
                             edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> const& chStatusToken,
                             edm::ESGetToken<CaloGeometry, CaloGeometryRecord> const& geoToken,
                             std::shared_ptr<PhiSymCache>& cache) const;
  void sumCache(PhiSymCache* streamCache, PhiSymCache const* cache) const;

  //---data memebers
  // available to derived classes
protected:
  mutable std::vector<DetId> barrelDetIds_;
  mutable std::vector<DetId> endcapDetIds_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geoToken_;
  edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> laserDbToken_;
  mutable edm::EDGetTokenT<EBRecHitCollection> ebToken_;
  mutable edm::EDGetTokenT<EBRecHitCollection> eeToken_;
  float etCutEB_;
  vector<double> eThresholdsEB_;
  float etCutEE_;
  vector<double> A_;
  vector<double> B_;
  float thrEEmod_;
  int nMisCalib_;
  int nSumEtValues_;
  vector<double> misCalibRangeEB_;
  float misCalibStepsEB_[11];
  vector<double> misCalibRangeEE_;
  float misCalibStepsEE_[11];
  //---geometry
  EcalRingCalibrationTools calibRing_;
  static const short kNRingsEB = EcalRingCalibrationTools::N_RING_BARREL;
  static const short kNRingsEE = EcalRingCalibrationTools::N_RING_ENDCAP;
  static const short ringsInOneEE = kNRingsEE / 2;
  mutable float etCutsEB_[kNRingsEB];
  mutable float etCutsEE_[kNRingsEE];
  mutable float eThresholdsEE_[kNRingsEE];
};

//----------IMPLEMENTATION----------------------------------------------------------------
EcalPhiSymRecHitProducerBase::EcalPhiSymRecHitProducerBase(const edm::ParameterSet& pSet, edm::ConsumesCollector&& cc)
    : geoToken_(cc.esConsumes()),
      laserDbToken_(cc.esConsumes()),
      ebToken_(cc.consumes<EBRecHitCollection>(pSet.getParameter<edm::InputTag>("barrelHitCollection"))),
      eeToken_(cc.consumes<EBRecHitCollection>(pSet.getParameter<edm::InputTag>("endcapHitCollection"))),
      etCutEB_(pSet.getParameter<double>("etCut_barrel")),
      eThresholdsEB_(pSet.getParameter<vector<double> >("eThresholds_barrel")),
      etCutEE_(pSet.getParameter<double>("etCut_endcap")),
      A_(pSet.getParameter<vector<double> >("A")),
      B_(pSet.getParameter<vector<double> >("B")),
      thrEEmod_(pSet.getParameter<double>("thrEEmod")),
      nMisCalib_(pSet.getParameter<int>("nMisCalib") / 2),
      nSumEtValues_(nMisCalib_ * 2 + 1),
      misCalibRangeEB_(pSet.getParameter<vector<double> >("misCalibRangeEB")),
      misCalibRangeEE_(pSet.getParameter<vector<double> >("misCalibRangeEE")) {}

void EcalPhiSymRecHitProducerBase::initializeJob() {
  //---set E thresholds, Et cuts and miscalib steps
  //---spectrum window: E > thr && Et < cut
  //---NOTE: etCutsEE need the geometry, so it is set later in beginLumi
  for (int iRing = 0; iRing < kNRingsEB; ++iRing)
    etCutsEB_[iRing] = -1;
  for (int iRing = 0; iRing < ringsInOneEE; ++iRing) {
    if (iRing < 30)
      eThresholdsEE_[iRing] = thrEEmod_ * (B_[0] + A_[0] * iRing) / 1000;
    else
      eThresholdsEE_[iRing] = thrEEmod_ * (B_[1] + A_[1] * iRing) / 1000;
    eThresholdsEE_[iRing + ringsInOneEE] = eThresholdsEE_[iRing];
    etCutsEE_[iRing] = -1;
    etCutsEE_[iRing + ringsInOneEE] = -1;
  }

  //---misCalib value init (nMisCalib is half oj the correct value!)
  float misCalibStepEB = fabs(misCalibRangeEB_[1] - misCalibRangeEB_[0]) / (nMisCalib_ * 2);
  float misCalibStepEE = fabs(misCalibRangeEE_[1] - misCalibRangeEE_[0]) / (nMisCalib_ * 2);
  for (int iMis = -nMisCalib_; iMis <= nMisCalib_; ++iMis) {
    //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
    int index = iMis > 0 ? iMis + nMisCalib_ : iMis == 0 ? 0 : iMis + nMisCalib_ + 1;
    misCalibStepsEB_[index] = iMis * misCalibStepEB;
    misCalibStepsEE_[index] = iMis * misCalibStepEE;
  }
}

void EcalPhiSymRecHitProducerBase::processEvent(edm::StreamID stream,
                                                edm::Event const& event,
                                                edm::EventSetup const& setup,
                                                PhiSymCache* streamCache) const {
  uint64_t totHitsEB = 0;
  uint64_t totHitsEE = 0;

  //---get recHits collections
  auto barrelRecHitsHandle = event.get(ebToken_);
  auto endcapRecHitsHandle = event.get(eeToken_);

  //---get the laser corrections
  edm::Timestamp evtTimeStamp(event.time().value());
  auto const& laser = setup.getData(laserDbToken_);

  //---get the geometry
  auto const& geometry = setup.getData(geoToken_);
  auto barrelGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  auto endcapGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

  //---EB---
  for (auto& recHit : barrelRecHitsHandle) {
    float energy = recHit.energy();
    EBDetId ebHit = EBDetId(recHit.id());
    int ring = calibRing_.getRingIndex(ebHit);
    //---if recHit energy is below thr even with the highest miscalib skip this recHit
    if (energy * misCalibRangeEB_[1] < eThresholdsEB_[ring])
      continue;
    float eta = barrelGeometry->getGeometry(ebHit)->getPosition().eta();

    //---compute et + miscalibration
    float etValues[nSumEtValues_];
    //---one can do this in one for loop from -nMis to +nMis but in this way the
    //---program is faster
    //---NOTE: nMisCalib is half on the value set in the cfg python
    etValues[0] = recHit.energy() / cosh(eta);
    for (int iMis = -nMisCalib_; iMis < 0; ++iMis) {
      //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
      int index = iMis + nMisCalib_ + 1;
      etValues[index] = etValues[0] * (1 + misCalibStepsEB_[index]);
      //---set et to zero if out of range [e_thr, et_thr+1]
      if (etValues[index] * cosh(eta) < eThresholdsEB_[ring] || etValues[index] > etCutsEB_[ring])
        etValues[index] = 0;
    }
    for (int iMis = 1; iMis <= nMisCalib_; ++iMis) {
      //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
      int index = iMis + nMisCalib_;
      etValues[index] = etValues[0] * (1 + misCalibStepsEB_[index]);
      //---set et to zero if out of range [e_thr, et_thr+1]
      if (etValues[index] * cosh(eta) < eThresholdsEB_[ring] || etValues[index] > etCutsEB_[ring])
        etValues[index] = 0;
    }
    //---set et to zero if out of range [e_thr, et_thr+1]
    if (energy < eThresholdsEB_[ring] || etValues[0] > etCutsEB_[ring])
      etValues[0] = 0;
    else
      ++totHitsEB;
    //---update the rechHit sumEt
    streamCache->recHitCollEB.at(ebHit.denseIndex())
        .AddHit(etValues, laser.getLaserCorrection(recHit.id(), evtTimeStamp));
  }

  //---EE---
  for (auto& recHit : endcapRecHitsHandle) {
    EEDetId eeHit = EEDetId(recHit.id());
    int ring = calibRing_.getRingIndex(eeHit) - kNRingsEB;
    float energy = recHit.energy();
    //---if recHit energy is below thr even with the highest miscalib skip this recHit
    if (energy * misCalibRangeEE_[1] < eThresholdsEE_[ring])
      continue;
    float eta = endcapGeometry->getGeometry(eeHit)->getPosition().eta();

    //---compute et + miscalibration
    float etValues[nSumEtValues_];
    //---one can do this in one for loop from -nMis to +nMis but in this way the
    //---program is faster
    //---NOTE: nMisCalib is half on the value set in the cfg python
    etValues[0] = recHit.energy() / cosh(eta);
    for (int iMis = -nMisCalib_; iMis < 0; ++iMis) {
      //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
      int index = iMis + nMisCalib_ + 1;
      etValues[index] = etValues[0] * (1 + misCalibStepsEE_[index]);
      //---set et to zero if out of range [e_thr, et_thr+1]
      if (etValues[index] * cosh(eta) < eThresholdsEE_[ring] || etValues[index] > etCutsEE_[ring])
        etValues[index] = 0;
    }
    for (int iMis = 1; iMis <= nMisCalib_; ++iMis) {
      //--- 0 -> 0; -i -> [1...n/2]; +i -> [n/2+1...n]
      int index = iMis + nMisCalib_;
      etValues[index] = etValues[0] * (1 + misCalibStepsEE_[index]);
      //---set et to zero if out of range [e_thr, et_thr+1]
      if (etValues[index] * cosh(eta) < eThresholdsEE_[ring] || etValues[index] > etCutsEE_[ring])
        etValues[index] = 0;
    }
    //---set et to zero if out of range [e_thr, et_thr+1]
    if (energy < eThresholdsEE_[ring] || etValues[0] > etCutsEE_[ring])
      etValues[0] = 0;
    else
      ++totHitsEE;
    //---update the rechHit sumEt
    streamCache->recHitCollEE.at(eeHit.denseIndex())
        .AddHit(etValues, laser.getLaserCorrection(recHit.id(), evtTimeStamp));
  }

  //---update the lumi info
  EcalPhiSymInfo thisEvent(totHitsEB, totHitsEE, 1, 0, 0, 0, 0);
  streamCache->ecalLumiInfo += thisEvent;
}

void EcalPhiSymRecHitProducerBase::initializeStreamCache(PhiSymCache* cache) const {
  //---Initialize the per-stream RecHitCollection
  //   both collections are initialized to contain the total
  //   number of crystals, ordered accrodingly to the hashedIndex.
  cache->clear();
  cache->recHitCollEB.resize(barrelDetIds_.size());
  cache->recHitCollEE.resize(endcapDetIds_.size());
  for (auto& ebDetId : barrelDetIds_) {
    EBDetId id(ebDetId);
    cache->recHitCollEB.at(id.denseIndex()) = EcalPhiSymRecHit(id.rawId(), nSumEtValues_);
  }
  for (auto& eeDetId : endcapDetIds_) {
    EEDetId id(eeDetId);
    cache->recHitCollEE.at(id.denseIndex()) = EcalPhiSymRecHit(id.rawId(), nSumEtValues_);
  }
}

void EcalPhiSymRecHitProducerBase::initializeGlobalCache(
    edm::EventSetup const& setup,
    edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> const& chStatusToken,
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> const& geoToken,
    std::shared_ptr<PhiSymCache>& cache) const {
  cache->clear();

  //---get the ecal geometry
  auto geometry = &setup.getData(geoToken);
  calibRing_.setCaloGeometry(geometry);

  //---get the channels status
  auto const& chStatus = setup.getData(chStatusToken);

  auto barrelGeometry = geometry->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  auto endcapGeometry = geometry->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
  barrelDetIds_ = barrelGeometry->getValidDetIds(DetId::Ecal, EcalBarrel);
  endcapDetIds_ = endcapGeometry->getValidDetIds(DetId::Ecal, EcalEndcap);
  cache->recHitCollEB.resize(barrelDetIds_.size());
  cache->recHitCollEE.resize(endcapDetIds_.size());
  for (auto& ebDetId : barrelDetIds_) {
    EBDetId id(ebDetId);
    ;
    cache->recHitCollEB.at(id.denseIndex()) =
        EcalPhiSymRecHit(ebDetId.rawId(), nSumEtValues_, chStatus[id].getStatusCode());
    int ring = calibRing_.getRingIndex(id);
    //---set etCut if first pass
    if (etCutsEB_[ring] == -1 && id.iphi() == 1) {
      auto cellGeometry = barrelGeometry->getGeometry(id);
      float eta = cellGeometry->getPosition().eta();
      etCutsEB_[ring] = eThresholdsEB_[ring] / cosh(eta) + etCutEB_;
    }
  }
  for (auto& eeDetId : endcapDetIds_) {
    EEDetId id(eeDetId);
    int ring = calibRing_.getRingIndex(id) - kNRingsEB;
    cache->recHitCollEE.at(id.denseIndex()) =
        EcalPhiSymRecHit(eeDetId.rawId(), nSumEtValues_, chStatus[id].getStatusCode());
    //---set eCutEE if first pass
    if (ring < ringsInOneEE && etCutsEE_[ring] == -1 && id.ix() == 50) {
      auto cellGeometry = endcapGeometry->getGeometry(id);
      etCutsEE_[ring] = eThresholdsEE_[ring] / cosh(cellGeometry->getPosition().eta()) + etCutEE_;
      etCutsEE_[ring + ringsInOneEE] = etCutsEE_[ring];
    }
  }
}

void EcalPhiSymRecHitProducerBase::sumCache(PhiSymCache* streamCache, PhiSymCache const* cache) const {
  //--- this could be improved.
  //    One has to make sure that the streamCache is the right hand argument
  //    in the sum to make sure that info like fillNumber and channel status
  //    are preserved since they are set in the global initialization and not
  //    in the stream one.
  cache->ecalLumiInfo += streamCache->ecalLumiInfo;

  for (unsigned int i = 0; i < cache->recHitCollEB.size(); ++i)
    cache->recHitCollEB.at(i) += streamCache->recHitCollEB.at(i);
  for (unsigned int i = 0; i < cache->recHitCollEE.size(); ++i)
    cache->recHitCollEE.at(i) += streamCache->recHitCollEE.at(i);
}

//****************************************************************************************
// Lumi producer
class EcalPhiSymRecHitProducerLumi : public edm::global::EDProducer<edm::StreamCache<PhiSymCache>,
                                                                    edm::LuminosityBlockCache<PhiSymCache>,
                                                                    edm::EndLuminosityBlockProducer,
                                                                    edm::Accumulator>,
                                     public EcalPhiSymRecHitProducerBase {
public:
  explicit EcalPhiSymRecHitProducerLumi(const edm::ParameterSet& pSet);
  ~EcalPhiSymRecHitProducerLumi(){};

private:
  //---methods
  // job
  void beginJob() { initializeJob(); };
  // lumi
  std::shared_ptr<PhiSymCache> globalBeginLuminosityBlock(edm::LuminosityBlock const& lumi,
                                                          edm::EventSetup const& setup) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) const override{};
  void globalEndLuminosityBlockProduce(edm::LuminosityBlock& lumi, edm::EventSetup const& setup) const override;
  // stream
  std::unique_ptr<PhiSymCache> beginStream(edm::StreamID stream) const override;
  void streamBeginLuminosityBlock(edm::StreamID stream,
                                  edm::LuminosityBlock const& lumi,
                                  edm::EventSetup const& setup) const override;
  void streamEndLuminosityBlock(edm::StreamID stream,
                                edm::LuminosityBlock const& lumi,
                                edm::EventSetup const& setup) const override;

  // event
  void accumulate(edm::StreamID stream, edm::Event const& event, edm::EventSetup const& setup) const;

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

std::shared_ptr<PhiSymCache> EcalPhiSymRecHitProducerLumi::globalBeginLuminosityBlock(
    edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) const {
  auto cache = std::make_shared<PhiSymCache>();

  //---Get LHC info
  const auto& lhcinfo = setup.getData(lhcInfoTokenLumi_);
  EcalPhiSymInfo thisLumi(0, 0, 0, 1, lhcinfo.fillNumber(), lhcinfo.delivLumi(), lhcinfo.recLumi());

  //---Reset global cache
  initializeGlobalCache(setup, chStatusTokenLumi_, geoTokenLumi_, cache);
  cache->ecalLumiInfo += thisLumi;

  return cache;
}

std::unique_ptr<PhiSymCache> EcalPhiSymRecHitProducerLumi::beginStream(edm::StreamID stream) const {
  //---create stream cache
  return std::make_unique<PhiSymCache>();
}

void EcalPhiSymRecHitProducerLumi::globalEndLuminosityBlockProduce(edm::LuminosityBlock& lumi,
                                                                   edm::EventSetup const& setup) const {
  auto cache = luminosityBlockCache(lumi.index());

  //---put the collections in the LuminosityBlocks tree
  auto ecalLumiInfo = std::make_unique<EcalPhiSymInfo>();
  *ecalLumiInfo = cache->ecalLumiInfo;
  auto recHitCollEB = std::make_unique<EcalPhiSymRecHitCollection>();
  *recHitCollEB = cache->recHitCollEB;
  auto recHitCollEE = std::make_unique<EcalPhiSymRecHitCollection>();
  *recHitCollEE = cache->recHitCollEE;

  lumi.put(std::move(ecalLumiInfo));
  lumi.put(std::move(recHitCollEB), "EB");
  lumi.put(std::move(recHitCollEE), "EE");
}

void EcalPhiSymRecHitProducerLumi::streamBeginLuminosityBlock(edm::StreamID stream,
                                                              edm::LuminosityBlock const& lumi,
                                                              edm::EventSetup const& setup) const {
  //---Reset cache
  initializeStreamCache(streamCache(stream));
}

void EcalPhiSymRecHitProducerLumi::streamEndLuminosityBlock(edm::StreamID stream,
                                                            edm::LuminosityBlock const& lumi,
                                                            edm::EventSetup const& setup) const {
  //---sum data from different streams into the common lumi cache
  sumCache(streamCache(stream), luminosityBlockCache(lumi.index()));
}

void EcalPhiSymRecHitProducerLumi::accumulate(edm::StreamID stream,
                                              edm::Event const& event,
                                              edm::EventSetup const& setup) const {
  processEvent(stream, event, setup, streamCache(stream));
}

//****************************************************************************************
// Run producer
class EcalPhiSymRecHitProducerRun : public edm::global::EDProducer<edm::StreamCache<PhiSymCache>,
                                                                   edm::RunCache<PhiSymCache>,
                                                                   edm::LuminosityBlockCache<PhiSymCache>,
                                                                   edm::EndRunProducer,
                                                                   edm::Accumulator>,
                                    public EcalPhiSymRecHitProducerBase {
public:
  explicit EcalPhiSymRecHitProducerRun(const edm::ParameterSet& pSet);
  ~EcalPhiSymRecHitProducerRun(){};

private:
  //---methods
  // job
  void beginJob() { initializeJob(); };
  // run
  std::shared_ptr<PhiSymCache> globalBeginRun(edm::Run const& run, edm::EventSetup const& setup) const override;
  void globalEndRun(edm::Run const& run, edm::EventSetup const& setup) const override{};
  void globalEndRunProduce(edm::Run& run, edm::EventSetup const& setup) const override;
  // lumi
  std::shared_ptr<PhiSymCache> globalBeginLuminosityBlock(edm::LuminosityBlock const& lumi,
                                                          edm::EventSetup const& setup) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) const override{};
  // stream
  std::unique_ptr<PhiSymCache> beginStream(edm::StreamID stream) const;
  void streamBeginRun(edm::StreamID stream, edm::Run const& run, edm::EventSetup const& setup) const override;
  void streamEndRun(edm::StreamID stream, edm::Run const& run, edm::EventSetup const& setup) const override;
  // event
  void accumulate(edm::StreamID stream, edm::Event const& event, edm::EventSetup const& setup) const;

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

std::shared_ptr<PhiSymCache> EcalPhiSymRecHitProducerRun::globalBeginRun(edm::Run const& run,
                                                                         edm::EventSetup const& setup) const {
  auto cache = std::make_shared<PhiSymCache>();
  initializeGlobalCache(setup, chStatusTokenRun_, geoTokenRun_, cache);
  return cache;
}

//---globalBeginLuminosityBlock is called just to correctly update the LHCInfo
std::shared_ptr<PhiSymCache> EcalPhiSymRecHitProducerRun::globalBeginLuminosityBlock(
    edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) const {
  //---Get LHC info
  //   LHCInfo only returns the correct luminosity information
  //   for each lumisection, accessing LHCInfo at the beginning
  //   of each run would return only the luminosity info of the
  //   first LS
  const auto& lhcinfo = setup.getData(lhcInfoTokenLumi_);
  EcalPhiSymInfo thisLumi(0, 0, 0, 1, lhcinfo.fillNumber(), lhcinfo.delivLumi(), lhcinfo.recLumi());

  runCache(lumi.getRun().index())->ecalLumiInfo += thisLumi;

  // dummy cache, it won't be used
  return std::make_shared<PhiSymCache>();
}

std::unique_ptr<PhiSymCache> EcalPhiSymRecHitProducerRun::beginStream(edm::StreamID stream) const {
  //---create stream cache
  return std::make_unique<PhiSymCache>();
}

void EcalPhiSymRecHitProducerRun::streamBeginRun(edm::StreamID stream,
                                                 edm::Run const& run,
                                                 edm::EventSetup const& setup) const {
  //---Reset global cache
  initializeStreamCache(streamCache(stream));
}

void EcalPhiSymRecHitProducerRun::streamEndRun(edm::StreamID stream,
                                               edm::Run const& run,
                                               edm::EventSetup const& setup) const {
  //---sum data from different streams into the common run cache
  sumCache(streamCache(stream), runCache(run.index()));
}

void EcalPhiSymRecHitProducerRun::globalEndRunProduce(edm::Run& run, edm::EventSetup const& setup) const {
  auto cache = runCache(run.index());

  //---put the collections in the Runs tree
  auto ecalLumiInfo = std::make_unique<EcalPhiSymInfo>();
  *ecalLumiInfo = cache->ecalLumiInfo;
  auto recHitCollEB = std::make_unique<EcalPhiSymRecHitCollection>();
  *recHitCollEB = cache->recHitCollEB;
  auto recHitCollEE = std::make_unique<EcalPhiSymRecHitCollection>();
  *recHitCollEE = cache->recHitCollEE;

  run.put(std::move(ecalLumiInfo));
  run.put(std::move(recHitCollEB), "EB");
  run.put(std::move(recHitCollEE), "EE");
}

void EcalPhiSymRecHitProducerRun::accumulate(edm::StreamID stream,
                                             edm::Event const& event,
                                             edm::EventSetup const& setup) const {
  processEvent(stream, event, setup, streamCache(stream));
}

DEFINE_FWK_MODULE(EcalPhiSymRecHitProducerLumi);
DEFINE_FWK_MODULE(EcalPhiSymRecHitProducerRun);

#endif
