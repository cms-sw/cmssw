#ifndef SiStripMonitorSummary_SiStripClassToMonitorCondData_h
#define SiStripMonitorSummary_SiStripClassToMonitorCondData_h
// -*- C++ -*-
//
// Package:     SiStripMonitorSummary
// Class  :     SiStripClassToMonitorCondData
//
// Original Author:  Evelyne Delmeire
// SiStripClassToMonitorCondData+SiStripCondDataMonitor -> SiStripMonitorCondData: Pieter David
//

// system include files
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"

class TkDetMap;
class SiStripPedestals;
class SiStripNoises;
class SiStripQuality;
class SiStripApvGain;
class SiStripLorentzAngle;
class SiStripBackPlaneCorrection;
class SiStripThreshold;
class SiStripDetCabling;

class SiStripPedestalsDQM;
class SiStripNoisesDQM;
class SiStripQualityDQM;
class SiStripApvGainsDQM;
class SiStripLorentzAngleDQM;
class SiStripBackPlaneCorrectionDQM;
class SiStripCablingDQM;
class SiStripThresholdDQM;

class SiStripClassToMonitorCondData {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  SiStripClassToMonitorCondData(edm::ParameterSet const &iConfig, edm::ConsumesCollector iC);
  ~SiStripClassToMonitorCondData();

  void beginRun(edm::RunNumber_t iRun, edm::EventSetup const &eSetup);
  void analyseCondData(const edm::EventSetup &);
  void end();
  void save();

  void getModMEsOnDemand(edm::EventSetup const &eSetup, uint32_t requestedDetId);
  void getLayerMEsOnDemand(edm::EventSetup const &eSetup,
                           std::string requestedSubDetector,
                           uint32_t requestedSide,
                           uint32_t requestedLayer);

private:
  edm::ParameterSet conf_;

  bool monitorPedestals_;
  bool monitorNoises_;
  bool monitorQuality_;
  bool monitorApvGains_;
  bool monitorLorentzAngle_;
  bool monitorBackPlaneCorrection_;
  bool monitorLowThreshold_;
  bool monitorHighThreshold_;
  bool monitorCabling_;

  std::unique_ptr<SiStripPedestalsDQM> pedestalsDQM_;
  std::unique_ptr<SiStripNoisesDQM> noisesDQM_;
  std::unique_ptr<SiStripQualityDQM> qualityDQM_;
  std::unique_ptr<SiStripApvGainsDQM> apvgainsDQM_;
  std::unique_ptr<SiStripLorentzAngleDQM> lorentzangleDQM_;
  std::unique_ptr<SiStripBackPlaneCorrectionDQM> bpcorrectionDQM_;
  std::unique_ptr<SiStripCablingDQM> cablingDQM_;
  std::unique_ptr<SiStripThresholdDQM> lowthresholdDQM_;
  std::unique_ptr<SiStripThresholdDQM> highthresholdDQM_;

  edm::ESGetToken<TkDetMap, TrackerTopologyRcd> tkDetMapToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoToken_;

  edm::ESGetToken<SiStripPedestals, SiStripPedestalsRcd> pedestalsToken_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noiseToken_;
  edm::ESGetToken<SiStripApvGain, SiStripApvGainSimRcd> simGainToken_;
  edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> gainTokenForNoise_;
  edm::ESGetToken<SiStripApvGain, SiStripApvGainRcd> gainToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleRcd> lorentzAngleToken_;
  edm::ESGetToken<SiStripBackPlaneCorrection, SiStripBackPlaneCorrectionRcd> backplaneCorrectionToken_;
  edm::ESGetToken<SiStripThreshold, SiStripThresholdRcd> thresholdToken_;
  edm::ESGetToken<SiStripDetCabling, SiStripDetCablingRcd> detCablingToken_;
};

#endif
