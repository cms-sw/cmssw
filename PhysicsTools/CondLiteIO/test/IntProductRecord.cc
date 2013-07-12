// -*- C++ -*-
//
// Package:     CondLiteIO
// Class  :     IntProductRecord
// 
// Implementation:
//     [Notes on implementation]
//
// Author:      Chris Jones
// Created:     Tue Jun 22 10:37:05 CDT 2010
// $Id$

#include "PhysicsTools/CondLiteIO/test/IntProductRecord.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"

EVENTSETUP_RECORD_REG(IntProductRecord);

TYPELOOKUP_DATA_REG(edmtest::IntProduct);

