// $Id: RPCTechTriggerConfig.h,v 1.1 2009/01/28 12:54:41 aosorio Exp $
#ifndef RPCTECHTRIGGERCONFIG_H 
#define RPCTECHTRIGGERCONFIG_H 1

// Include files
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

};
#endif // RPCTECHTRIGGERCONFIG_H
