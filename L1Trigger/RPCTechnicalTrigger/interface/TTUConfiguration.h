// $Id: TTUConfiguration.h,v 1.4 2012/05/15 08:06:25 eulisse Exp $
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

class TTUConfiguration {
public: 
  virtual ~TTUConfiguration() {}
  virtual bool initialise( int , int )=0;
  
  virtual void preprocess(TTUInput &)=0;
  
  TTULogicUnit  * m_ttulogic;

  const TTUBoardSpecs * m_ttuboardspecs;
  
  TTUBoardSpecs::TTUBoardConfig * m_ttuconf;
  
protected:
  
private:
  
};
#endif // INTERFACE_TTUCONFIGURATION_H
