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
  RBCConfiguration():m_rbcboardspecs(nullptr) {}
  RBCConfiguration(const RBCBoardSpecs * rbcspecs);
  RBCConfiguration(const char * _logic);

  RBCConfiguration(RBCConfiguration&&) = default;
  RBCConfiguration& operator=(RBCConfiguration&&) = default;

  virtual ~RBCConfiguration() = default;
  virtual bool initialise()=0;

  virtual void preprocess(RBCInput &)=0;
  
  RBCLogicUnit* rbclogic() { return m_rbclogic.get();}
  
protected:
  const RBCBoardSpecs * m_rbcboardspecs;
  std::unique_ptr<RBCLogicUnit>  m_rbclogic;
  
  //RBCBoardSpecs::RBCBoardConfig * m_rbcconf;
  
  
private:

};
#endif // INTERFACE_RBCCONFIGURATION_H
