// -*- C++ -*-
//
// Package:     L1Trigger/RPCTechnicalTrigger
// Class  :     TTUConfiguration
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 16 Nov 2018 19:02:33 GMT
//

// system include files

// user include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUConfiguration.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TTUConfiguration::TTUConfiguration(const char* logic):
  m_ttuboardspecs{nullptr},
  m_ttulogic{ logic }
{
}

TTUConfiguration::TTUConfiguration( const TTUBoardSpecs* ttuspecs):
  m_ttuboardspecs{ttuspecs},
  m_ttulogic{}
{}
