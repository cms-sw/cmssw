// $Id: RBCProcessTestSignal.h,v 1.1 2009/05/16 19:43:30 aosorio Exp $
#ifndef RBCPROCESSTESTSIGNAL_H 
#define RBCPROCESSTESTSIGNAL_H 1


// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>

/** @class RBCProcessTestSignal RBCProcessTestSignal.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-10
 */
class RBCProcessTestSignal : public ProcessInputSignal {
public: 
  /// Standard constructor
  RBCProcessTestSignal( ) {}; 
  
  RBCProcessTestSignal( const char * ); 
  
  virtual ~RBCProcessTestSignal( ); ///< Destructor
  
  int  next();
  
  void rewind();
  
  void showfirst();
  
  RPCInputSignal * retrievedata() { 
    return  m_lbin; 
  };
  
protected:
  
private:
  
  std::ifstream * m_in;
  
  RBCInput * m_input;

  RPCInputSignal * m_lbin;
  
  
};
#endif // RBCPROCESSTESTSIGNAL_H
