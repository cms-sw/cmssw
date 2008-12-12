/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"

DEFINE_SEAL_MODULE();


#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxMipRcd.h"
REGISTER_PLUGIN(SiStripDeDxMipRcd, PhysicsTools::Calibration::HistogramD2D);





