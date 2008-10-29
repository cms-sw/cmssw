#ifndef FWCore_MessageLogger_AbstractMLscribe_h
#define FWCore_MessageLogger_AbstractMLscribe_h

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"

namespace edm  {
namespace service {       

class AbstractMLscribe 
{
public:
  // ---  birth/death:
  AbstractMLscribe();
  virtual ~AbstractMLscribe();

  // ---  methods needed for logging
  virtual
  void  runCommand(MessageLoggerQ::OpCode  opcode, void * operand);

private:
  // --- no copying:
  AbstractMLscribe(AbstractMLscribe const &);
  void  operator = (AbstractMLscribe const &);

};  // AbstractMLscribe

}   // end of namespace service
}  // namespace edm


#endif // FWCore_MessageLogger_AbstractMLscribe_h
