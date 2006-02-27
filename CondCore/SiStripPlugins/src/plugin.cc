/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripControlCabling.h"
#include "CondFormats/DataRecord/interface/SiStripReadoutCablingRcd.h"
#include "CondFormats/DataRecord/interface/SiStripControlCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"


DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(SiStripReadoutCablingRcd, SiStripReadoutCabling);
REGISTER_PLUGIN(SiStripControlCablingRcd, SiStripControlCabling);
REGISTER_PLUGIN(SiStripPedestalsRcd,SiStripPedestals);
REGISTER_PLUGIN(SiStripNoisesRcd,SiStripNoises);
