// -*- C++ -*-
//
// Package:     CMSSW/CondFormats/Alignment
// Class  :     T_EventSetup_OpticalAlignments.cc
// 
// Implementation:
//     create all the 'infrastructure' needed to get into the Context
//
// Author:      Chris Jones
// Created:     Mon Apr 18 16:42:52 EDT 2005
// $Id: T_EventSetup_OpticalAlignments.cc,v 1.6 2006/11/07 22:23:43 arce Exp $
//

// system include files

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
