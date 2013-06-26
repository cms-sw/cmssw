// $Id: TTUTwoORLogic.h,v 1.1 2009/06/17 15:27:24 aosorio Exp $
#ifndef INTERFACE_TTUTWOORLOGIC_H 
#define INTERFACE_TTUTWOORLOGIC_H 1

// Include files

#include "L1Trigger/RPCTechnicalTrigger/interface/TTULogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"

#include "L1Trigger/RPCTechnicalTrigger/interface/TTUTrackingAlg.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUSectorORLogic.h"

#include <iostream>
#include <vector>

/** @class TTUTwoORLogic TTUTwoORLogic.h interface/TTUTwoORLogic.h
 *  
 *
 *  @author Andres Felipe Osorio Oliveros
 *  @date   2009-06-16
 */

class TTUTwoORLogic : public TTULogic {
public: 
  /// Standard constructor
  TTUTwoORLogic( ); 

  virtual ~TTUTwoORLogic( ); ///< Destructor

  //... from TTULogic interface:
  
  bool process( const TTUInput & );
  
  void setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & );
  
  //...

protected:

private:

  bool m_debug;

  TTUTrackingAlg * m_ttuLogic;
  
  TTUSectorORLogic * m_rbcLogic;
  

};
#endif // INTERFACE_TTUTWOORLOGIC_H
