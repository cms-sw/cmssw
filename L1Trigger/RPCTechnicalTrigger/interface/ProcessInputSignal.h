// $Id: ProcessInputSignal.h,v 1.1 2009/01/30 15:42:47 aosorio Exp $
#ifndef INTERFACE_PROCESSINPUTSIGNAL_H 
#define INTERFACE_PROCESSINPUTSIGNAL_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"

#include <map>

/** @class ProcessInputSignal ProcessInputSignal.h interface/ProcessInputSignal.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-10
 */
class ProcessInputSignal {
public: 
  
  virtual ~ProcessInputSignal() {};
  
  virtual int  next()  = 0;
  
  virtual RPCInputSignal * retrievedata() = 0;
  
protected:
  
private:
  
};
#endif // INTERFACE_PROCESSINPUTSIGNAL_H
