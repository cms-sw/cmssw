// $Id: TTULogicUnit.h,v 1.5 2009/08/09 11:11:36 aosorio Exp $
#ifndef TTULOGICUNIT_H 
#define TTULOGICUNIT_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/LogicTool.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCLogicUnit.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTULogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"

#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"

#include <bitset>

/** @class TTULogicUnit TTULogicUnit.h
 *  
 *
 *  @author Andres Osorio
 *  @date   2008-10-25
 */

class TTULogicUnit : public RPCLogicUnit {
public: 
  /// Standard constructor
  TTULogicUnit( );
  
  TTULogicUnit( const char * );
  
  virtual ~TTULogicUnit( ); ///< Destructor

  bool initialise();
  
  void setlogic( const char * );
  
  void setBoardSpecs( const TTUBoardSpecs::TTUBoardConfig & );
  
  void run( const TTUInput & );

  void run( const TTUInput & , int );
  
  bool isTriggered() {
    return m_logic->m_triggersignal;
  };
    
protected:
  
private:
  
  std::string m_logtype;
  
  TTULogic             * m_logic;
  
  LogicTool<TTULogic>  * m_logtool;

  bool m_debug;
    
};
#endif // TTUPAC_H
