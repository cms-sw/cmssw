// $Id: RBCLogicUnit.h,v 1.2 2009/05/08 10:24:04 aosorio Exp $
#ifndef RBCLOGICUNIT_H 
#define RBCLOGICUNIT_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/LogicTool.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCLogicUnit.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"

#include <bitset>

/** @class RBCLogicUnit RBCLogicUnit.h
 *  
 *
 *  @author Andres Osorio
 *  @date   2008-10-25
 */

class RBCLogicUnit : public RPCLogicUnit {
public: 
  /// Standard constructor
  RBCLogicUnit( );
  
  RBCLogicUnit( const char * );
  
  virtual ~RBCLogicUnit( ); ///< Destructor
  
  bool initialise();
  
  void setlogic( const char * );
  
  void run( const RBCInput & , std::bitset<2> & );
  
  std::bitset<6> * getlayersignal(int _idx) { return m_layersignal[_idx]; };

  bool isTriggered() {
    return m_logic->m_triggersignal;
  };

protected:
  
private:

  std::string m_logtype;
  
  std::bitset<6> * m_layersignal[2];

  RBCLogic            * m_logic;
  
  LogicTool<RBCLogic> * m_logtool;

  bool m_debug;
    
};
#endif // RBCLOGICUNIT_H
