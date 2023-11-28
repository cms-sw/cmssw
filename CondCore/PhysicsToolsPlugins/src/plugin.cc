/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/PhysicsToolsObjects/interface/Histogram2D.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram3D.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxMipRcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxMip_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxPion_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxKaon_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxElectron_3D_Rcd.h"

#include "CondFormats/DataRecord/interface/PhysicsTFormulaPayloadRcd.h"
#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTFormulaPayload.h"
#include "CondFormats/DataRecord/interface/PhysicsTGraphPayloadRcd.h"
#include "CondFormats/PhysicsToolsObjects/interface/PhysicsTGraphPayload.h"

#include "CondFormats/DataRecord/interface/DropBoxMetadataRcd.h"
#include "CondFormats/Common/interface/DropBoxMetadata.h"

#include "CondCore/CondDB/interface/Serialization.h"

REGISTER_PLUGIN(DropBoxMetadataRcd, DropBoxMetadata);

REGISTER_PLUGIN(SiStripDeDxMipRcd, PhysicsTools::Calibration::HistogramD2D);
REGISTER_PLUGIN(SiStripDeDxMip_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);
REGISTER_PLUGIN_NO_SERIAL(SiStripDeDxProton_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);
REGISTER_PLUGIN_NO_SERIAL(SiStripDeDxPion_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);
REGISTER_PLUGIN_NO_SERIAL(SiStripDeDxKaon_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);
REGISTER_PLUGIN_NO_SERIAL(SiStripDeDxElectron_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);
REGISTER_PLUGIN(PhysicsTFormulaPayloadRcd, PhysicsTFormulaPayload);
REGISTER_PLUGIN(PhysicsTGraphPayloadRcd, PhysicsTGraphPayload);
