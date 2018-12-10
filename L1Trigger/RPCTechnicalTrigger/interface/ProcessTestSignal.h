#ifndef PROCESSTESTSIGNAL_H 
#define PROCESSTESTSIGNAL_H 1

// Include files

#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h" 
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RPCData.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ios>
#include <cmath>
#include <vector>
#include <memory>

/** @class ProcessTestSignal ProcessTestSignal.h
 *  
 * 
 *
 *
 *
 *  @author Andres Osorio
 *  @date   2008-11-14
 */
class ProcessTestSignal : public ProcessInputSignal {
public: 
  explicit ProcessTestSignal( const char * );
  
  ~ProcessTestSignal( ) override; ///< Destructor
  
  int  next() override;
  
  void rewind();
  
  void showfirst();
  
  void reset();
  
  RPCInputSignal * retrievedata() override {
    return  m_lbin.get();
  };
  
  void mask() {};
  void force() {};
  
protected:
  
private:
  
  void builddata();
  
  std::ifstream  m_in;
  
  std::unique_ptr<RPCInputSignal> m_lbin;
  
  std::vector<std::unique_ptr<RPCData>> m_vecdata;
  
  std::map<int,RBCInput*> m_data;
  
  
};
#endif // PROCESSTESTSIGNAL_H
