/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"

DEFINE_SEAL_MODULE();


#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
REGISTER_PLUGIN(SiStripPedestalsRcd,SiStripPedestals);

#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
REGISTER_PLUGIN(SiStripNoisesRcd,SiStripNoises);

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
REGISTER_PLUGIN(SiStripFedCablingRcd,SiStripFedCabling);

#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
REGISTER_PLUGIN(SiStripLorentzAngleRcd,SiStripLorentzAngle);

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
REGISTER_PLUGIN(SiStripApvGainRcd,SiStripApvGain);

#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"
#include "CondFormats/DataRecord/interface/SiStripBadStripRcd.h"
REGISTER_PLUGIN(SiStripBadStripRcd,SiStripBadStrip);
#include "CondFormats/DataRecord/interface/SiStripBadModuleRcd.h"
REGISTER_PLUGIN(SiStripBadModuleRcd,SiStripBadStrip);
#include "CondFormats/DataRecord/interface/SiStripBadFiberRcd.h"
REGISTER_PLUGIN(SiStripBadFiberRcd,SiStripBadStrip);
#include "CondFormats/DataRecord/interface/SiStripBadChannelRcd.h"
REGISTER_PLUGIN(SiStripBadChannelRcd,SiStripBadStrip);

#include "CondFormats/SiStripObjects/interface/SiStripModuleHV.h"
#include "CondFormats/DataRecord/interface/SiStripModuleHVRcd.h"
REGISTER_PLUGIN(SiStripModuleHVRcd,SiStripModuleHV);

#include "CondFormats/SiStripObjects/interface/SiStripRunSummary.h"
#include "CondFormats/DataRecord/interface/SiStripRunSummaryRcd.h"
REGISTER_PLUGIN(SiStripRunSummaryRcd,SiStripRunSummary);

#include "CondFormats/SiStripObjects/interface/SiStripPerformanceSummary.h"
#include "CondFormats/DataRecord/interface/SiStripPerformanceSummaryRcd.h"
REGISTER_PLUGIN(SiStripPerformanceSummaryRcd,SiStripPerformanceSummary);

#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"
REGISTER_PLUGIN(SiStripThresholdRcd,SiStripThreshold);

#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "CondFormats/DataRecord/interface/SiStripSummaryRcd.h"
REGISTER_PLUGIN(SiStripSummaryRcd,SiStripSummary);

#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxMipRcd.h"
REGISTER_PLUGIN(SiStripDeDxMipRcd, PhysicsTools::Calibration::HistogramD2D);





