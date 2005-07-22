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
// $Id: T_Context_Pedestals.cc,v 1.1 2005/04/23 17:51:45 innocent Exp $
//

// system include files

// user include files
#include "CondFormats/Alignment/interface/Alignments.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

//using Alignments;
CONTEXT_DATA_REG(Alignments);
