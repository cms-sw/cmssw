//-------------------------------------------------
//
/**  \class DTConfigTSTheta
 *
 *   Configurable parameters and constants 
 *   for Level-1 Muon DT Trigger - TS Theta
 *
 *
 *   \author c. Battilana
 *
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_TSTHETA_H
#define DT_CONFIG_TSTHETA_H

//---------------
// C++ Headers --
//---------------

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/DTUtilities/interface/DTConfig.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigTSTheta : public DTConfig {

  public:

/*   //! Constants: first and last step to start trigger finding */
/*   static const int NSTEPL=24, NSTEPF=9; */
  
  //! Constants: number of cell (BTI) in theta view planes
  static const int NCELLTH=57;

  //! Constructor  
  DTConfigTSTheta(edm::ParameterSet& ps);

  //! Destructor
  ~DTConfigTSTheta();

  //! Returns the debug flag
  inline bool debug() const { return m_debug; }

  //! print the setup
  void print() const ;

  //! return pointer to parameter set
  const edm::ParameterSet* getParameterSet() { return m_ps; }

  private:

  //! load pset values into class variables
  void setDefaults();

  edm::ParameterSet* m_ps;

  bool m_debug;

};

#endif
