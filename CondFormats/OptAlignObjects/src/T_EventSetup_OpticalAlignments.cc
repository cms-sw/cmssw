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
// $Id: T_EventSetup_OpticalAlignments.cc,v 1.1 2006/01/26 13:44:14 case Exp $
//

// system include files

// user include files
#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/CSCZSensors.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

//using Alignments;
EVENTSETUP_DATA_REG(OpticalAlignments);
EVENTSETUP_DATA_REG(CSCZSensors);
