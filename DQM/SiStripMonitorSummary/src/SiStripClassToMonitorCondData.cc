// -*- C++ -*-
//
// Package:     SiStripMonitorSummary
// Class  :     SiStripClassToMonitorCondData
//
// Original Author:  Evelyne Delmeire
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripClassToMonitorCondData.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripApvGainsDQM.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripBackPlaneCorrectionDQM.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripCablingDQM.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripLorentzAngleDQM.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripNoisesDQM.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripPedestalsDQM.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripQualityDQM.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripThresholdDQM.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"

#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

// std
#include <cstdlib>
#include <algorithm>

SiStripClassToMonitorCondData::SiStripClassToMonitorCondData(edm::ParameterSet const& iConfig,
                                                             edm::ConsumesCollector iC)
    : conf_(iConfig) {
  monitorPedestals_ = iConfig.getParameter<bool>("MonitorSiStripPedestal");
  monitorNoises_ = iConfig.getParameter<bool>("MonitorSiStripNoise");
  monitorQuality_ = iConfig.getParameter<bool>("MonitorSiStripQuality");
  monitorApvGains_ = iConfig.getParameter<bool>("MonitorSiStripApvGain");
  monitorLorentzAngle_ = iConfig.getParameter<bool>("MonitorSiStripLorentzAngle");
  monitorBackPlaneCorrection_ = iConfig.getParameter<bool>("MonitorSiStripBackPlaneCorrection");
  monitorLowThreshold_ = iConfig.getParameter<bool>("MonitorSiStripLowThreshold");
  monitorHighThreshold_ = iConfig.getParameter<bool>("MonitorSiStripHighThreshold");
  monitorCabling_ = iConfig.getParameter<bool>("MonitorSiStripCabling");
  tkDetMapToken_ = iC.esConsumes<edm::Transition::BeginRun>();
  tTopoToken_ = iC.esConsumes<edm::Transition::BeginRun>();
  if (monitorPedestals_) {
    pedestalsToken_ = iC.esConsumes();
  }
  if (monitorNoises_) {
    noiseToken_ = iC.esConsumes();
    const auto& hPSet = iConfig.getParameter<edm::ParameterSet>("SiStripNoisesDQM_PSet");
    if (hPSet.getParameter<bool>("SimGainRenormalisation")) {
      simGainToken_ = iC.esConsumes<edm::Transition::BeginRun>();
    } else if (hPSet.getParameter<bool>("GainRenormalisation")) {
      gainTokenForNoise_ = iC.esConsumes<edm::Transition::BeginRun>();
    }
  }
  if (monitorQuality_) {
    const auto& fPSet = conf_.getParameter<edm::ParameterSet>("FillConditions_PSet");
    const auto& qualityLabel = fPSet.getParameter<std::string>("StripQualityLabel");
    qualityToken_ = iC.esConsumes(edm::ESInputTag{"", qualityLabel});
  }
  if (monitorApvGains_) {
    gainToken_ = iC.esConsumes();
  }
  if (monitorLorentzAngle_) {
    lorentzAngleToken_ = iC.esConsumes();
  }
  if (monitorBackPlaneCorrection_) {
    backplaneCorrectionToken_ = iC.esConsumes();
  }
  if (monitorLowThreshold_ || monitorHighThreshold_) {
    thresholdToken_ = iC.esConsumes();
  }
  if (monitorCabling_) {
    detCablingToken_ = iC.esConsumes();
  }
}

SiStripClassToMonitorCondData::~SiStripClassToMonitorCondData() {}

void SiStripClassToMonitorCondData::beginRun(edm::RunNumber_t iRun, edm::EventSetup const& eSetup) {
  const auto tTopo = &eSetup.getData(tTopoToken_);
  const auto& fPSet = conf_.getParameter<edm::ParameterSet>("FillConditions_PSet");
  const TkDetMap* tkDetMap{nullptr};
  if (fPSet.getParameter<bool>("HistoMaps_On")) {
    tkDetMap = &eSetup.getData(tkDetMapToken_);
  }
  if (monitorPedestals_) {
    const auto& hPSet = conf_.getParameter<edm::ParameterSet>("SiStripPedestalsDQM_PSet");
    pedestalsDQM_ = std::make_unique<SiStripPedestalsDQM>(pedestalsToken_, iRun, hPSet, fPSet, tTopo, tkDetMap);
  }

  if (monitorNoises_) {
    const auto& hPSet = conf_.getParameter<edm::ParameterSet>("SiStripNoisesDQM_PSet");
    const SiStripApvGain* gain = nullptr;
    if (hPSet.getParameter<bool>("SimGainRenormalisation")) {
      gain = &eSetup.getData(simGainToken_);
    } else if (hPSet.getParameter<bool>("GainRenormalisation")) {
      gain = &eSetup.getData(gainTokenForNoise_);
    }
    noisesDQM_ = std::make_unique<SiStripNoisesDQM>(noiseToken_, iRun, hPSet, fPSet, tTopo, tkDetMap, gain);
  }

  if (monitorQuality_) {
    const auto& hPSet = conf_.getParameter<edm::ParameterSet>("SiStripQualityDQM_PSet");
    qualityDQM_ = std::make_unique<SiStripQualityDQM>(qualityToken_, iRun, hPSet, fPSet, tTopo, tkDetMap);
  }

  if (monitorApvGains_) {
    const auto& hPSet = conf_.getParameter<edm::ParameterSet>("SiStripApvGainsDQM_PSet");
    apvgainsDQM_ = std::make_unique<SiStripApvGainsDQM>(gainToken_, iRun, hPSet, fPSet, tTopo, tkDetMap);
  }

  if (monitorLorentzAngle_) {
    const auto& hPSet = conf_.getParameter<edm::ParameterSet>("SiStripLorentzAngleDQM_PSet");
    lorentzangleDQM_ =
        std::make_unique<SiStripLorentzAngleDQM>(lorentzAngleToken_, iRun, hPSet, fPSet, tTopo, tkDetMap);
  }

  if (monitorBackPlaneCorrection_) {
    const auto& hPSet = conf_.getParameter<edm::ParameterSet>("SiStripBackPlaneCorrectionDQM_PSet");
    bpcorrectionDQM_ =
        std::make_unique<SiStripBackPlaneCorrectionDQM>(backplaneCorrectionToken_, iRun, hPSet, fPSet, tTopo, tkDetMap);
  }

  if (monitorLowThreshold_) {
    const auto& hPSet = conf_.getParameter<edm::ParameterSet>("SiStripLowThresholdDQM_PSet");
    lowthresholdDQM_ = std::make_unique<SiStripThresholdDQM>(thresholdToken_, iRun, hPSet, fPSet, tTopo, tkDetMap);
  }

  if (monitorHighThreshold_) {
    const auto& hPSet = conf_.getParameter<edm::ParameterSet>("SiStripHighThresholdDQM_PSet");
    highthresholdDQM_ = std::make_unique<SiStripThresholdDQM>(thresholdToken_, iRun, hPSet, fPSet, tTopo, tkDetMap);
  }

  if (monitorCabling_) {
    const auto& hPSet = conf_.getParameter<edm::ParameterSet>("SiStripCablingDQM_PSet");
    cablingDQM_ = std::make_unique<SiStripCablingDQM>(detCablingToken_, iRun, hPSet, fPSet, tTopo, tkDetMap);
  }
}

void SiStripClassToMonitorCondData::getModMEsOnDemand(edm::EventSetup const& eSetup, uint32_t requestedDetId) {
  if (monitorPedestals_) {
    pedestalsDQM_->analysisOnDemand(eSetup, requestedDetId);
  }
  if (monitorNoises_) {
    noisesDQM_->analysisOnDemand(eSetup, requestedDetId);
  }
  if (monitorQuality_) {
    qualityDQM_->analysisOnDemand(eSetup, requestedDetId);
    qualityDQM_->fillGrandSummaryMEs();
  }  // fillGrand. for SiStripquality
  if (monitorApvGains_) {
    apvgainsDQM_->analysisOnDemand(eSetup, requestedDetId);
  }
  if (monitorLorentzAngle_) {
    lorentzangleDQM_->analysisOnDemand(eSetup, requestedDetId);
  }
  if (monitorBackPlaneCorrection_) {
    bpcorrectionDQM_->analysisOnDemand(eSetup, requestedDetId);
  }
  if (monitorCabling_) {
    cablingDQM_->analysisOnDemand(eSetup, requestedDetId);
  }
  if (monitorLowThreshold_) {
    lowthresholdDQM_->analysisOnDemand(eSetup, requestedDetId);
  }
  if (monitorHighThreshold_) {
    highthresholdDQM_->analysisOnDemand(eSetup, requestedDetId);
  }
}
// -----

//
// ----- getlayerMEsOnDemand
//
void SiStripClassToMonitorCondData::getLayerMEsOnDemand(edm::EventSetup const& eSetup,
                                                        std::string requestedSubDetector,
                                                        uint32_t requestedSide,
                                                        uint32_t requestedLayer) {
  if (monitorPedestals_) {
    pedestalsDQM_->analysisOnDemand(eSetup, requestedSubDetector, requestedSide, requestedLayer);
  }
  if (monitorNoises_) {
    noisesDQM_->analysisOnDemand(eSetup, requestedSubDetector, requestedSide, requestedLayer);
  }
  if (monitorQuality_) {
    qualityDQM_->analysisOnDemand(eSetup, requestedSubDetector, requestedSide, requestedLayer);
    qualityDQM_->fillGrandSummaryMEs();
  }
  if (monitorApvGains_) {
    apvgainsDQM_->analysisOnDemand(eSetup, requestedSubDetector, requestedSide, requestedLayer);
  }
  if (monitorLorentzAngle_) {
    lorentzangleDQM_->analysisOnDemand(eSetup, requestedSubDetector, requestedSide, requestedLayer);
  }
  if (monitorBackPlaneCorrection_) {
    bpcorrectionDQM_->analysisOnDemand(eSetup, requestedSubDetector, requestedSide, requestedLayer);
  }
  if (monitorCabling_) {
    cablingDQM_->analysisOnDemand(eSetup, requestedSubDetector, requestedSide, requestedLayer);
  }
  if (monitorLowThreshold_) {
    lowthresholdDQM_->analysisOnDemand(eSetup, requestedSubDetector, requestedSide, requestedLayer);
  }
  if (monitorHighThreshold_) {
    highthresholdDQM_->analysisOnDemand(eSetup, requestedSubDetector, requestedSide, requestedLayer);
  }
}

//
// ----- Analyze
//
void SiStripClassToMonitorCondData::analyseCondData(edm::EventSetup const& eSetup) {
  if (monitorPedestals_) {
    pedestalsDQM_->analysis(eSetup);
  }
  if (monitorNoises_) {
    noisesDQM_->analysis(eSetup);
  }
  if (monitorQuality_) {
    qualityDQM_->analysis(eSetup);
    qualityDQM_->fillGrandSummaryMEs();
  }  // fillGrand. for SiStripquality
  if (monitorApvGains_) {
    apvgainsDQM_->analysis(eSetup);
  }
  if (monitorLorentzAngle_) {
    lorentzangleDQM_->analysis(eSetup);
  }
  if (monitorBackPlaneCorrection_) {
    bpcorrectionDQM_->analysis(eSetup);
  }
  if (monitorCabling_) {
    cablingDQM_->analysis(eSetup);
  }
  if (monitorLowThreshold_) {
    lowthresholdDQM_->analysis(eSetup);
  }
  if (monitorHighThreshold_) {
    highthresholdDQM_->analysis(eSetup);
  }

}  // analyze
// -----

void SiStripClassToMonitorCondData::end() {
  if (monitorPedestals_) {
    pedestalsDQM_->end();
  }
  if (monitorNoises_) {
    noisesDQM_->end();
  }
  if (monitorLowThreshold_) {
    lowthresholdDQM_->end();
  }
  if (monitorHighThreshold_) {
    highthresholdDQM_->end();
  }
  if (monitorApvGains_) {
    apvgainsDQM_->end();
  }
  if (monitorLorentzAngle_) {
    lorentzangleDQM_->end();
  }
  if (monitorBackPlaneCorrection_) {
    bpcorrectionDQM_->end();
  }
  if (monitorQuality_) {
    qualityDQM_->end();
  }
  if (monitorCabling_) {
    cablingDQM_->end();
  }
}

void SiStripClassToMonitorCondData::save() {
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");

  DQMStore* dqmStore_ = edm::Service<DQMStore>().operator->();

  if (outputMEsInRootFile) {
    dqmStore_->save(outputFileName);
  }
}
