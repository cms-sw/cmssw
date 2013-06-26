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
// $Id: T_EventSetup_OpticalAlignments.cc,v 1.8 2010/02/20 20:58:08 wmtan Exp $
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
#include "FWCore/Utilities/interface/typelookup.h"

//using Alignments;
TYPELOOKUP_DATA_REG(OpticalAlignments);
TYPELOOKUP_DATA_REG(CSCZSensors);
TYPELOOKUP_DATA_REG(CSCRSensors);
TYPELOOKUP_DATA_REG(MBAChBenchCalPlate);
TYPELOOKUP_DATA_REG(MBAChBenchSurveyPlate);
TYPELOOKUP_DATA_REG(Inclinometers);
TYPELOOKUP_DATA_REG(PXsensors);
