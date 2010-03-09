// -*- C++ -*-
//
// Package:     test
// Class  :     ValueExample
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Sep  5 19:52:01 EDT 2005
//

#include "FWCore/Integration/test/ValueExample.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

ValueExample::ValueExample(const edm::ParameterSet& iPSet):
value_(iPSet.getParameter<int>("value"))
{
}

ValueExample::~ValueExample()
{
}

void ValueExample::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<int>("value");
  descriptions.addDefault(desc);
}
