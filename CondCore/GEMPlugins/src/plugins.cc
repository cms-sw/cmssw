/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Sven Dildick --TAMU
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/GEMObjects/interface/GEMELMap.h"
#include "CondFormats/DataRecord/interface/GEMELMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMMaskedStrips.h"
#include "CondFormats/DataRecord/interface/GEMMaskedStripsRcd.h"
#include "CondFormats/GEMObjects/interface/GEMDeadStrips.h"
#include "CondFormats/DataRecord/interface/GEMDeadStripsRcd.h"
REGISTER_PLUGIN(GEMELMapRcd,GEMELMap);
REGISTER_PLUGIN(GEMMaskedStripsRcd, GEMMaskedStrips);
REGISTER_PLUGIN(GEMDeadStripsRcd, GEMDeadStrips);

