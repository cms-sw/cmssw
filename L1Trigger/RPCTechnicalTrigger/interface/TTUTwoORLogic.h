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

  //... from TTULogic interface:
  
  bool process( const TTUInput & ) override;
  
  void setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & ) override;
  
  //...

protected:

private:


  TTUTrackingAlg m_ttuLogic;
  
  TTUSectorORLogic m_rbcLogic;

  bool m_debug;
  

};
#endif // INTERFACE_TTUTWOORLOGIC_H
