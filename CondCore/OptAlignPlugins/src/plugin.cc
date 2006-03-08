/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/CSCZSensors.h"
#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h"
#include "CondFormats/DataRecord/interface/CSCZSensorsRcd.h"
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(OpticalAlignmentsRcd,OpticalAlignments);
REGISTER_PLUGIN(CSCZSensorsRcd,CSCZSensors);
