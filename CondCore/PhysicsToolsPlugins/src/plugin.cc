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
REGISTER_PLUGIN(SiStripDeDxMipRcd, PhysicsTools::Calibration::HistogramD2D);
#include "CondFormats/DataRecord/interface/SiStripDeDxMip_3D_Rcd.h"
REGISTER_PLUGIN(SiStripDeDxMip_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_3D_Rcd.h"
REGISTER_PLUGIN(SiStripDeDxProton_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);
#include "CondFormats/DataRecord/interface/SiStripDeDxPion_3D_Rcd.h"
REGISTER_PLUGIN(SiStripDeDxPion_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);
#include "CondFormats/DataRecord/interface/SiStripDeDxKaon_3D_Rcd.h"
REGISTER_PLUGIN(SiStripDeDxKaon_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);
#include "CondFormats/DataRecord/interface/SiStripDeDxElectron_3D_Rcd.h"
REGISTER_PLUGIN(SiStripDeDxElectron_3D_Rcd, PhysicsTools::Calibration::HistogramD3D);


#include "CondFormats/DataRecord/interface/DropBoxMetadataRcd.h"
#include "CondFormats/Common/interface/DropBoxMetadata.h"


REGISTER_PLUGIN(DropBoxMetadataRcd,DropBoxMetadata);







