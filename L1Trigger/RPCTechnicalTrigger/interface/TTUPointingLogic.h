// $Id: TTUPointingLogic.h,v 1.1 2009/08/09 11:11:36 aosorio Exp $
#ifndef TTUPOINTINGLOGIC_H 
#define TTUPOINTINGLOGIC_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTULogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"

#include "L1Trigger/RPCTechnicalTrigger/interface/TTUWedgeORLogic.h"

#include <iostream>
#include <vector>

/** @class TTUPointingLogic TTUPointingLogic.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2009-07-29
 */

class TTUPointingLogic : public TTULogic {
public: 
  /// Standard constructor
  TTUPointingLogic( ); 
  
  virtual ~TTUPointingLogic( ); ///< Destructor

  //... from TTULogic interface:
  
  bool process( const TTUInput & );
  
  void setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & );
  
  //...
  
protected:
  
private:

  bool m_debug;

  TTUWedgeORLogic * m_wedgeLogic;
    
    
};
#endif // TTUPOINTINGLOGIC_H
