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
#include<iostream>

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
  DTConfigTrigUnit() {};

  //! Destructor
  ~DTConfigTrigUnit();

  //! Debug flag
  inline bool debug() const { return m_debug; }

  //! Digi-to-bti-input offset 500 (tdc units) --> wait to solve with Nicola Amapane
  inline int MCDigiOffset() const { return m_digioffset; }

  //! Minicrate "fine" sincronization parameter [0,25] ns
  inline int MCSetupTime() const { return  m_setuptime; }

  //! Print the setup
  void print() const ;

 /*  //! Return pointer to parameter set */
/*   const edm::ParameterSet* getParameterSet() { return m_ps; } */

  //! Set debug flag
  inline void setDebug(bool debug) { m_debug=debug; }

  private:

  //! Load pset values into class variables
  void setDefaults(const edm::ParameterSet& m_ps);

  //const edm::ParameterSet* m_ps;

  bool m_debug;
  int m_digioffset;
  int m_setuptime;

};

#endif
