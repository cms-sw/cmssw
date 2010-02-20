/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "DPGAnalysis/SiStripTools/interface/APVLatency.h"
#include "DPGAnalysis/SiStripTools/interface/APVLatencyRcd.h"


REGISTER_PLUGIN(APVLatencyRcd,APVLatency);
