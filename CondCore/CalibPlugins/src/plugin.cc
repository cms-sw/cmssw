/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/Calibration/interface/Pedestals.h"
#include "CondFormats/DataRecord/interface/PedestalsRcd.h"
#include "CondFormats/Calibration/interface/mySiStripNoises.h"
#include "CondFormats/DataRecord/interface/mySiStripNoisesRcd.h"
#include "CondFormats/Calibration/interface/Efficiency.h"
#include "CondFormats/DataRecord/interface/ExEfficiency.h"
#include "CondFormats/Common/interface/BaseKeyed.h"
#include "CondCore/IOVService/interface/KeyListProxy.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(PedestalsRcd,Pedestals);
REGISTER_PLUGIN(anotherPedestalsRcd,Pedestals);
REGISTER_PLUGIN(mySiStripNoisesRcd,mySiStripNoises);
REGISTER_PLUGIN(ExEfficiencyRcd, condex::Efficiency);
REGISTER_PLUGIN(ExDwarfRcd, cond::BaseKeyed);
// REGISTER_PLUGIN(ExDwarfListRcd, cond::KeyList);
REGISTER_KEYLIST_PLUGIN(ExDwarfListRcd, cond::KeyList, ExDwarfRcd);
