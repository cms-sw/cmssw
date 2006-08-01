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
#include "CondFormats/OptAlignObjects/interface/CSCRSensors.h"
#include "CondFormats/OptAlignObjects/interface/Inclinometers.h"
#include "CondFormats/OptAlignObjects/interface/PX_sensors.h"

#include "CondFormats/OptAlignObjects/interface/MBAChBenchCalPlate.h"
#include "CondFormats/OptAlignObjects/interface/MBAChBenchSurveyPlate.h"
#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h"
#include "CondFormats/DataRecord/interface/CSCZSensorsRcd.h"
#include "CondFormats/DataRecord/interface/CSCRSensorsRcd.h"
#include "CondFormats/DataRecord/interface/InclinometersRcd.h"
#include "CondFormats/DataRecord/interface/PX_sensorsRcd.h"

#include "CondFormats/DataRecord/interface/MBAChBenchCalPlateRcd.h"
#include "CondFormats/DataRecord/interface/MBAChBenchSurveyPlateRcd.h"
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(OpticalAlignmentsRcd,OpticalAlignments);
REGISTER_PLUGIN(CSCZSensorsRcd,CSCZSensors);
REGISTER_PLUGIN(CSCRSensorsRcd,CSCRSensors);
REGISTER_PLUGIN(MBAChBenchCalPlateRcd,MBAChBenchCalPlate);
REGISTER_PLUGIN(MBAChBenchSurveyPlateRcd,MBAChBenchSurveyPlate);
REGISTER_PLUGIN(InclinometersRcd, Inclinometers);
REGISTER_PLUGIN(PX_sensorsRcd, PX_sensors);
