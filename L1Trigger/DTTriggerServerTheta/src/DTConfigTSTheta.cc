//-------------------------------------------------
//
//   Class: DTConfigTSTheta
//
//   Description: Configurable parameters and constants 
//   for Level1 Mu DT Trigger - TS Theta 
//
//
//   Author List:
//   C. Battilana
//
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTTriggerServerTheta/interface/DTConfigTSTheta.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//----------------
// Constructors --
//----------------
DTConfigTSTheta::DTConfigTSTheta(edm::ParameterSet& ps) : m_ps(&ps) { 

  setDefaults();

}

//--------------
// Destructor --
//--------------
DTConfigTSTheta::~DTConfigTSTheta() {}

//--------------
// Operations --
//--------------

void
DTConfigTSTheta::setDefaults() {

  // Debug flag 
  m_debug = m_ps->getUntrackedParameter<bool>("Debug");

}

void 
DTConfigTSTheta::print() const {

  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : TSTheta chips                       *" << std::endl;
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                                                                            *" << std::endl;
  std::cout << "Debug flag : " <<  m_debug << std::endl;
  std::cout << "******************************************************************************" << std::endl;

}

