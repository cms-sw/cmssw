// $Id: RBCConfiguration.h,v 1.2 2012/05/11 06:01:08 eulisse Exp $
#ifndef INTERFACE_RBCCONFIGURATION_H 
#define INTERFACE_RBCCONFIGURATION_H 1

// Include files
#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"

#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLogicUnit.h"

/** @class RBCConfiguration RBCConfiguration.h interface/RBCConfiguration.h
 *  
 *
 *  Configuration Interface: Deals with configuration of RBC hardware
 *  
 *
 *  @author Andres Osorio
 *  @date   2008-10-29
 */

class RBCConfiguration {
public: 
  virtual ~RBCConfiguration() {}
  virtual bool initialise()=0;

  virtual void preprocess(RBCInput &)=0;
    
  RBCLogicUnit  * m_rbclogic;
  
  const RBCBoardSpecs * m_rbcboardspecs;
  
  RBCBoardSpecs::RBCBoardConfig * m_rbcconf;
  
protected:
  
private:

};
#endif // INTERFACE_RBCCONFIGURATION_H
