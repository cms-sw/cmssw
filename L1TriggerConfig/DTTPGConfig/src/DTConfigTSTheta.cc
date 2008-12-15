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
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTSTheta.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//----------------
// Constructors --
//----------------
DTConfigTSTheta::DTConfigTSTheta(const edm::ParameterSet& ps) { 

  setDefaults(ps);

}

DTConfigTSTheta::DTConfigTSTheta() : m_debug(false) {

}

//--------------
// Destructor --
//--------------
DTConfigTSTheta::~DTConfigTSTheta() {}

//--------------
// Operations --
//--------------

void
DTConfigTSTheta::setDefaults(const edm::ParameterSet& ps) {

  // Debug flag 
  m_debug = ps.getUntrackedParameter<bool>("Debug");

}

void 
DTConfigTSTheta::print() const {

  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : TSTheta chips                       *" << std::endl;
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*                                                                            *" << std::endl;
  std::cout << "Debug flag : " <<  debug() << std::endl;
  std::cout << "******************************************************************************" << std::endl;

}

