// $Id: $
#ifndef RPCPROCESSTESTSIGNAL_H 
#define RPCPROCESSTESTSIGNAL_H 1

// Include files

#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h" 
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCData.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>
#include <vector>


/** @class RPCProcessTestSignal RPCProcessTestSignal.h
 *  
 * 
 *
 *
 *
 *  @author Andres Osorio
 *  @date   2008-11-14
 */
class RPCProcessTestSignal : public ProcessInputSignal {
public: 
  /// Standard constructor
  RPCProcessTestSignal( ) { }; 
  
  RPCProcessTestSignal( const char * );
  
  virtual ~RPCProcessTestSignal( ); ///< Destructor
  
  int  next();
  
  void rewind();
  
  void showfirst();
  
  void reset();
  
  void mask() {};
  
  void force() {};
  
  RPCInputSignal * retrievedata() {
    return  m_lbin;
  };
  
protected:
  
private:
  
  void builddata();
  
  std::ifstream * m_in;
  
  RPCData  * m_block;
  
  RBCInput * m_rbcinput;
  
  RPCInputSignal * m_lbin;
  
  std::vector<RPCData*> m_vecdata;
  
  std::map<int,RBCInput*> m_data;
  
  
};
#endif // RPCPROCESSTESTSIGNAL_H
