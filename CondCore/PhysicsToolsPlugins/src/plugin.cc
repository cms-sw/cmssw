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
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_2D_Rcd.h"
REGISTER_PLUGIN(SiStripDeDxProton_2D_Rcd, PhysicsTools::Calibration::VHistogramD2D);
#include "CondFormats/DataRecord/interface/SiStripDeDxPion_2D_Rcd.h"
REGISTER_PLUGIN(SiStripDeDxPion_2D_Rcd, PhysicsTools::Calibration::VHistogramD2D);
#include "CondFormats/DataRecord/interface/SiStripDeDxKaon_2D_Rcd.h"
REGISTER_PLUGIN(SiStripDeDxKaon_2D_Rcd, PhysicsTools::Calibration::VHistogramD2D);
#include "CondFormats/DataRecord/interface/SiStripDeDxElectron_2D_Rcd.h"
REGISTER_PLUGIN(SiStripDeDxElectron_2D_Rcd, PhysicsTools::Calibration::VHistogramD2D);






