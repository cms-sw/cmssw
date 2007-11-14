
/*
 *   File: DataFormats/Scalers/src/LumiScalers.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/LumiScalers.h"

LumiScalers::LumiScalers() : m_data(0) 
{ 
}

LumiScalers::LumiScalers(uint16_t rawData)
{ }

LumiScalers::~LumiScalers() { } 


/// Pretty-print operator for LumiScalers
std::ostream& operator<<(std::ostream& s, const LumiScalers& c) 
{
  s << " LumiScalers: ";
  return s;
}
