/*
 *  plugin.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 7/24/05.
 *
 */

#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/CSCZSensors.h"
#include "CondFormats/OptAlignObjects/interface/CSCRSensors.h"
#include "CondFormats/OptAlignObjects/interface/Inclinometers.h"
#include "CondFormats/OptAlignObjects/interface/PXsensors.h"

#include "CondFormats/OptAlignObjects/interface/MBAChBenchCalPlate.h"
#include "CondFormats/OptAlignObjects/interface/MBAChBenchSurveyPlate.h"
#include "CondFormats/DataRecord/interface/OpticalAlignmentsRcd.h"
#include "CondFormats/DataRecord/interface/CSCZSensorsRcd.h"
#include "CondFormats/DataRecord/interface/CSCRSensorsRcd.h"
#include "CondFormats/DataRecord/interface/InclinometersRcd.h"
#include "CondFormats/DataRecord/interface/PXsensorsRcd.h"

#include "CondFormats/DataRecord/interface/MBAChBenchCalPlateRcd.h"
#include "CondFormats/DataRecord/interface/MBAChBenchSurveyPlateRcd.h"

REGISTER_PLUGIN(OpticalAlignmentsRcd,OpticalAlignments);
REGISTER_PLUGIN(CSCZSensorsRcd,CSCZSensors);
REGISTER_PLUGIN(CSCRSensorsRcd,CSCRSensors);
REGISTER_PLUGIN(MBAChBenchCalPlateRcd,MBAChBenchCalPlate);
REGISTER_PLUGIN(MBAChBenchSurveyPlateRcd,MBAChBenchSurveyPlate);
REGISTER_PLUGIN(InclinometersRcd, Inclinometers);
REGISTER_PLUGIN(PXsensorsRcd, PXsensors);
