// user include files
#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/CSCZSensors.h"
#include "CondFormats/OptAlignObjects/interface/CSCRSensors.h"
#include "CondFormats/OptAlignObjects/interface/MBAChBenchCalPlate.h"
#include "CondFormats/OptAlignObjects/interface/MBAChBenchSurveyPlate.h"
#include "CondFormats/OptAlignObjects/interface/Inclinometers.h"
#include "CondFormats/OptAlignObjects/interface/PXsensors.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

//using Alignments;
EVENTSETUP_DATA_REG(OpticalAlignments);
EVENTSETUP_DATA_REG(CSCZSensors);
EVENTSETUP_DATA_REG(CSCRSensors);
EVENTSETUP_DATA_REG(MBAChBenchCalPlate);
EVENTSETUP_DATA_REG(MBAChBenchSurveyPlate);
EVENTSETUP_DATA_REG(Inclinometers);
EVENTSETUP_DATA_REG(PXsensors);