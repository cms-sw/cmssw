//-------------------------------------------------
//
//   Class: DTConfigTrigUnit
//
//   Description: Configurable parameters and constants
//   for Level1 Mu DT Trigger - Trigger Unit (DT chamber MiniCrate)
//
//
//   Author List:
//   S. Vanini
//
//-----------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTrigUnit.h"

//---------------
// C++ Headers --
//---------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//----------------
// Constructors --
//----------------
DTConfigTrigUnit::DTConfigTrigUnit(const edm::ParameterSet& ps) {
  setDefaults(ps);
  if (debug())
    print();
}

//--------------
// Destructor --
//--------------
DTConfigTrigUnit::~DTConfigTrigUnit() {}

//--------------
// Operations --
//--------------

void DTConfigTrigUnit::setDefaults(const edm::ParameterSet& m_ps) {
  // Debug flag
  m_debug = m_ps.getUntrackedParameter<bool>("Debug");
}

void DTConfigTrigUnit::print() const {
  std::cout << "******************************************************************************" << std::endl;
  std::cout << "*              DTTrigger configuration : Trigger Unit parameters             *" << std::endl;
  std::cout << "******************************************************************************" << std::endl
            << std::endl;
  std::cout << "Debug flag : " << debug() << std::endl;
  std::cout << "******************************************************************************" << std::endl;
}
