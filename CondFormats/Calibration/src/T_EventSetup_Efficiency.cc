// -*- C++ -*-
//
// Package:     EDMProto
// Class  :     T_Context_Pedestals
// 
// Implementation:
//     create all the 'infrastructure' needed to get into the Context
//
// Author:      Chris Jones
// Created:     Mon Apr 18 16:42:52 EDT 2005
// $Id: T_EventSetup_Efficiency.cc,v 1.2 2008/07/12 15:07:17 innocent Exp $
//

// system include files

// user include files
#include "CondFormats/Calibration/interface/Efficiency.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

EVENTSETUP_DATA_REG(condex::Efficiency);
