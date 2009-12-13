#include "DataFormats/L1Trigger/interface/L1TriggerError.h"


L1TriggerError::L1TriggerError(unsigned code) :
  code_(code)
{
}


L1TriggerError::~L1TriggerError()
{
}


unsigned L1TriggerError::prodID() { 
  return (code_>>16) & 0xffff; 
}


unsigned L1TriggerError::prodErr() { 
  return code_ & 0xffff; 
}
