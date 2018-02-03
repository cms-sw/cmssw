/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Sven Dildick --TAMU
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "CondFormats/DataRecord/interface/GEMEMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMMaskedStrips.h"
#include "CondFormats/DataRecord/interface/GEMMaskedStripsRcd.h"
#include "CondFormats/GEMObjects/interface/GEMDeadStrips.h"
#include "CondFormats/DataRecord/interface/GEMDeadStripsRcd.h"
REGISTER_PLUGIN(GEMEMapRcd,GEMEMap);
REGISTER_PLUGIN(GEMMaskedStripsRcd, GEMMaskedStrips);
REGISTER_PLUGIN(GEMDeadStripsRcd, GEMDeadStrips);

#include "CondFormats/GEMObjects/interface/ME0EMap.h"
#include "CondFormats/DataRecord/interface/ME0EMapRcd.h"
#include "CondFormats/GEMObjects/interface/ME0MaskedStrips.h"
#include "CondFormats/DataRecord/interface/ME0MaskedStripsRcd.h"
#include "CondFormats/GEMObjects/interface/ME0DeadStrips.h"
#include "CondFormats/DataRecord/interface/ME0DeadStripsRcd.h"
REGISTER_PLUGIN(ME0EMapRcd,ME0EMap);
REGISTER_PLUGIN(ME0MaskedStripsRcd, ME0MaskedStrips);
REGISTER_PLUGIN(ME0DeadStripsRcd, ME0DeadStrips);
