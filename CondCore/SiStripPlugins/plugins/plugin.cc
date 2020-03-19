/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
REGISTER_PLUGIN(SiStripPedestalsRcd, SiStripPedestals);

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
REGISTER_PLUGIN(SiStripNoisesRcd, SiStripNoises);

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
REGISTER_PLUGIN(SiStripFedCablingRcd, SiStripFedCabling);

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
REGISTER_PLUGIN(SiStripLorentzAngleRcd, SiStripLorentzAngle);
REGISTER_PLUGIN(SiStripLorentzAngleSimRcd, SiStripLorentzAngle);

#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"
REGISTER_PLUGIN(SiStripBackPlaneCorrectionRcd, SiStripBackPlaneCorrection);

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
REGISTER_PLUGIN(SiStripApvGainRcd, SiStripApvGain);
REGISTER_PLUGIN(SiStripApvGain2Rcd, SiStripApvGain);
REGISTER_PLUGIN(SiStripApvGain3Rcd, SiStripApvGain);
REGISTER_PLUGIN(SiStripApvGainSimRcd, SiStripApvGain);

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
REGISTER_PLUGIN(SiStripBadStripRcd, SiStripBadStrip);
REGISTER_PLUGIN(SiStripBadModuleRcd, SiStripBadStrip);
REGISTER_PLUGIN(SiStripBadFiberRcd, SiStripBadStrip);
REGISTER_PLUGIN(SiStripBadChannelRcd, SiStripBadStrip);
REGISTER_PLUGIN(SiStripDCSStatusRcd, SiStripBadStrip);

#include "CondFormats/SiStripObjects/interface/SiStripDetVOff.h"
REGISTER_PLUGIN(SiStripDetVOffRcd, SiStripDetVOff);

#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
REGISTER_PLUGIN(SiStripLatencyRcd, SiStripLatency);

#include "CondFormats/SiStripObjects/interface/SiStripBaseDelay.h"
REGISTER_PLUGIN(SiStripBaseDelayRcd, SiStripBaseDelay);

#include "CondFormats/SiStripObjects/interface/SiStripRunSummary.h"
REGISTER_PLUGIN(SiStripRunSummaryRcd, SiStripRunSummary);

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
REGISTER_PLUGIN(SiStripThresholdRcd, SiStripThreshold);
REGISTER_PLUGIN(SiStripClusterThresholdRcd, SiStripThreshold);

#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
REGISTER_PLUGIN(SiStripSummaryRcd, SiStripSummary);

#include "CondFormats/SiStripObjects/interface/SiStripConfObject.h"
REGISTER_PLUGIN(SiStripConfObjectRcd, SiStripConfObject);

#include "CondFormats/SiStripObjects/interface/SiStripApvSimulationParameters.h"
namespace {
  struct initializeApvSimulationParameters {
    void operator()(SiStripApvSimulationParameters& param) { param.calculateIntegrals(); }
  };
}  // namespace
REGISTER_PLUGIN_INIT(SiStripApvSimulationParametersRcd,
                     SiStripApvSimulationParameters,
                     initializeApvSimulationParameters);

#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
namespace {
  struct initializeCabling {
    void operator()(Phase2TrackerCabling& c) { c.initializeCabling(); }
  };
}  // namespace
REGISTER_PLUGIN_INIT(Phase2TrackerCablingRcd, Phase2TrackerCabling, initializeCabling);
