// -*- C++ -*-
//
// Package:     L1Trigger/RPCTechnicalTrigger
// Class  :     RBCConfiguration
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 15 Nov 2018 21:02:24 GMT
//

// system include files

// user include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCConfiguration.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RBCConfiguration::RBCConfiguration(const RBCBoardSpecs * rbcspecs):
  m_rbcboardspecs{rbcspecs},
  m_rbclogic{std::make_unique<RBCLogicUnit>()}
{
}

RBCConfiguration::RBCConfiguration(const char * _logic):
  m_rbcboardspecs{nullptr},
  m_rbclogic{std::make_unique<RBCLogicUnit>( _logic )}
{}

