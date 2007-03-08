#ifndef RPCTechnicalTrigger_RBCPolicy_h
#define RPCTechnicalTrigger_RBCPolicy_h
#include <iostream>
#include <string>
class RBCLogic;
class RBCPolicy{
 public:
  enum Policy{Patterns, ChamberOR};
  RBCPolicy(Policy p);
  virtual ~RBCPolicy();
  std::string message();
  RBCLogic*   instance();
  
 private:
  Policy pol;
  RBCLogic* l;
};
#endif
