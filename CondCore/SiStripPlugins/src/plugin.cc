/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/TrackerMapping/interface/SiStripReadOutCabling.h"
#include "CondFormats/TrackerMapping/interface/SiStripControlCabling.h"
#include "CondFormats/DataRecord/interface/SiStripReadOutCablingRcd.h"
#include "CondFormats/DataRecord/interface/SiStripControlCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"


DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(SiStripReadOutCablingRcd, SiStripReadOutCabling);
REGISTER_PLUGIN(SiStripControlCablingRcd, SiStripControlCabling);
REGISTER_PLUGIN(SiStripPedestalsRcd,SiStripPedestals);
