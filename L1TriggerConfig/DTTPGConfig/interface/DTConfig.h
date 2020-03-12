//-------------------------------------------------
//
/**  \class DTConfig
 *
 *   Configurable common parameters 
 *   for Level-1 Muon DT Trigger 
 *
 *   \author  S. Vanini
 *
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_H
#define DT_CONFIG_H

//---------------
// C++ Headers --
//---------------

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfig {
public:
  //! Constants: first and last step to start trigger finding
  static const int NSTEPL = 24, NSTEPF = 9;

  static const int NBTITC = 4;

  //! Constant: number of TRACOs in input to a TSS
  static const int NTCTSS = 4;

  //! Constructor
  DTConfig(){};

  //! Destructor
  virtual ~DTConfig(){};
};

#endif
