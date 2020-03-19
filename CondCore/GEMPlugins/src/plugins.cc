/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Sven Dildick --TAMU
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/GEMObjects/interface/GEMELMap.h"
#include "CondFormats/DataRecord/interface/GEMELMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "CondFormats/DataRecord/interface/GEMeMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMMaskedStrips.h"
#include "CondFormats/DataRecord/interface/GEMMaskedStripsRcd.h"
#include "CondFormats/GEMObjects/interface/GEMDeadStrips.h"
#include "CondFormats/DataRecord/interface/GEMDeadStripsRcd.h"
REGISTER_PLUGIN(GEMELMapRcd, GEMELMap);
REGISTER_PLUGIN(GEMeMapRcd, GEMeMap);
REGISTER_PLUGIN(GEMMaskedStripsRcd, GEMMaskedStrips);
REGISTER_PLUGIN(GEMDeadStripsRcd, GEMDeadStrips);
