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
// $Id: IntProductRecord.cc,v 1.1 2010/06/22 21:51:17 chrjones Exp $

#include "PhysicsTools/CondLiteIO/test/IntProductRecord.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"

EVENTSETUP_RECORD_REG(IntProductRecord);

TYPELOOKUP_DATA_REG(edmtest::IntProduct);

