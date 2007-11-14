
/*
 *   File: DataFormats/Scalers/src/L1TriggerScalers.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/L1TriggerScalers.h"

L1TriggerScalers::L1TriggerScalers() : m_data(0) 
{ 
}

L1TriggerScalers::L1TriggerScalers(uint16_t rawData)
{ }

L1TriggerScalers::~L1TriggerScalers() { } 


/// Pretty-print operator for L1TriggerScalers
std::ostream& operator<<(std::ostream& s, const L1TriggerScalers& c) 
{
  s << " L1TriggerScalers: ";
  return s;
}
