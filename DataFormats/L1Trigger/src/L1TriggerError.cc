#include "DataFormats/L1Trigger/interface/L1TriggerError.h"

L1TriggerError::L1TriggerError(unsigned short prod, unsigned short code) : code_(prod << 16 & code) {}

L1TriggerError::~L1TriggerError() {}

unsigned L1TriggerError::prodID() { return (code_ >> 16) & 0xffff; }

unsigned L1TriggerError::prodErr() { return code_ & 0xffff; }
