//-------------------------------------------------
//
/**  \class DTConfigTrigUnit
 *
 *   Configurable parameters and constants 
 *   for Level-1 Muon DT Trigger - Trigger Unit
 *
 *
 *   \author S. Vanini
 *
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_TrigUnit_H
#define DT_CONFIG_TrigUnit_H

//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigTrigUnit : public DTConfig {
public:
  //! Constructor
  DTConfigTrigUnit(const edm::ParameterSet& ps);

  //! Constructor
  DTConfigTrigUnit(){};

  //! Destructor
  ~DTConfigTrigUnit() override;

  //! Debug flag
  inline bool debug() const { return m_debug; }

  //! Print the setup
  void print() const;

  //! Set debug flag
  inline void setDebug(bool debug) { m_debug = debug; }

private:
  //! Load pset values into class variables
  void setDefaults(const edm::ParameterSet& m_ps);

  bool m_debug;
};

#endif
