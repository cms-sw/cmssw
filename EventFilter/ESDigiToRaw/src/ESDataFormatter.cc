
#include <iostream> 
#include <sstream>
#include <iomanip> 

#include "EventFilter/ESDigiToRaw/interface/ESDataFormatter.h"

using namespace std; 
using namespace edm; 



string ESDataFormatter::print(const  Word64 & word) const
{
  ostringstream str;
  if (printInHex_)
    str << "Word64:  0x" << setw(16) << setfill('0') << hex << (word) << dec ;
  else 
    str << "Word64:  " << reinterpret_cast<const bitset<64>&> (word);
  return str.str();
}

string ESDataFormatter::print(const  Word16 & word) const
{
  ostringstream str;
  if (printInHex_) 
    str << "Word16:  0x" << setw(8) << setfill('0') << hex << (word) << dec ;
  else 
    str << "Word16:  " << reinterpret_cast<const bitset<16>&> (word);
  return str.str();
}

