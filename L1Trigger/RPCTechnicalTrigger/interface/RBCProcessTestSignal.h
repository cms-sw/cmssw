#ifndef RBCPROCESSTESTSIGNAL_H 
#define RBCPROCESSTESTSIGNAL_H 1


// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>
#include <memory>

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
  explicit RBCProcessTestSignal( const char * ); 
  
  ~RBCProcessTestSignal( ) override; ///< Destructor
  
  int  next() override;
  
  void rewind();
  
  void showfirst();
  
  RPCInputSignal * retrievedata() override { 
    return  m_lbin.get(); 
  };
  
protected:
  
private:
  
  std::ifstream m_in;
  
  RBCInput m_input;

  std::unique_ptr<RPCInputSignal> m_lbin;
  
  
};
#endif // RBCPROCESSTESTSIGNAL_H
