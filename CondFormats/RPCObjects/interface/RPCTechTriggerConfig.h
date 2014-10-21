// $Id: $
#ifndef RPCTECHTRIGGERCONFIG_H 
#define RPCTECHTRIGGERCONFIG_H 1

// Include files
#include "CondFormats/Serialization/interface/Serializable.h"

#include <iostream>

/** @class RPCTechTriggerConfig RPCTechTriggerConfig.h
 *  
 *   This class describes the basic database configuration object
 *
 *  @author Andres Osorio
 *  @date   2008-12-07
 */
class RPCTechTriggerConfig {
public: 
  /// Standard constructor
  RPCTechTriggerConfig( ) {
    m_runId       = -1;
    m_runType     = -1;
    m_triggerMode = -1;
    
  }; 
  
  RPCTechTriggerConfig( int run, int runtype, int trigmode ) {
    m_runId       = run;
    m_runType     = runtype;
    m_triggerMode = trigmode;
    
  };
  
  
  virtual ~RPCTechTriggerConfig( ) {}; ///< Destructor
  
  int m_runId;
  int m_runType;
  int m_triggerMode;
  
protected:

private:


  COND_SERIALIZABLE;
};
#endif // RPCTECHTRIGGERCONFIG_H
