//-------------------------------------------------
//
/**  \class DTConfigBti
 *
 *   Configurable parameters and constants 
 *   for Level-1 Muon DT Trigger - Time Pedestals
 *
 *   \authors:  C.Battilana, M.Meneghelli
 *
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_PEDESTALS_H
#define DT_CONFIG_PEDESTALS_H

//---------------
// C++ Headers --
//---------------

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------


class DTConfigPedestals : DTConfig {

  public:
 
  //! Default Constructor
  DTConfigPedestals();

  //! Destructor 
  ~DTConfigPedestals();

  //! Get wire by wire delay
  float getOffset(const DTWireId& wire);

  //! Set parameters from ES
  void setES(DTTPGParameters const *tpgParams,
	     DTT0 const *t0Params = 0);

  //! Set t0i subtraction
  void setUseT0 (bool useT0) { my_useT0 = useT0; }
 
  //! Set debug flag
  void setDebug (bool debug) { my_debug = debug; }

  //! Print the setup
  void print() const ;

 private :

  //! Debug flag
  inline int debug() const { return my_debug; }

  //! Use t0i
  inline bool useT0() const { return my_useT0; }

 private :

  bool my_debug;
  bool my_useT0;
  DTTPGParameters const * my_tpgParams;  
  DTT0 const *my_t0i;              // pointed object not owned by this class

};

#endif
