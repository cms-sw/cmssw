#ifndef RPCTechnicalTrigger_RBCPolicy_h
#define RPCTechnicalTrigger_RBCPolicy_h

/**  \class RBCPolicy
 *
 *  \author M. Maggi, C. Viviani, D. Pagano - University of Pavia & INFN Pavia
 *
 *
 */


#include <iostream>
#include <string>


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}


class RBCLogic;
class RBCPolicy{

 public:
 

  enum Policy{Neighbours, Patterns, ChamberOR};
  
  RBCPolicy(Policy p, const edm::ParameterSet&);
  
  virtual ~RBCPolicy();
  
  std::string message();
  
  RBCLogic*   instance();
  
  Policy pol;
  
 private:
  RBCLogic* l;
  bool neighbours; 
  int BX;
  int  majority;
  int poly;
    
  
};
#endif
