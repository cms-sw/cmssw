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


DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(PedestalsRcd,Pedestals);
