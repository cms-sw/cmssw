#ifndef INTERFACE_TTUCONFIGURATION_H 
#define INTERFACE_TTUCONFIGURATION_H 1

// Include files
#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"

#include "L1Trigger/RPCTechnicalTrigger/interface/TTUInput.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTULogicUnit.h"

/** @class TTUConfiguration TTUConfiguration.h interface/TTUConfiguration.h
 *  
 *
 *  Configuration Interface: Deals with configuration of TTU hardware
 * 
 *
 *  @author Andres Osorio
 *  @date   2008-10-29
 */
#include <memory>

class TTUConfiguration {
public: 
  TTUConfiguration( const char*);
  TTUConfiguration( const TTUBoardSpecs*);
  virtual ~TTUConfiguration() = default;
  virtual bool initialise( int , int )=0;
  
  virtual void preprocess(TTUInput &)=0;

  TTULogicUnit* ttulogic() { return &m_ttulogic; }

  const TTUBoardSpecs * m_ttuboardspecs;
  
protected:
  
private:
  TTULogicUnit  m_ttulogic;
  
};
#endif // INTERFACE_TTUCONFIGURATION_H
