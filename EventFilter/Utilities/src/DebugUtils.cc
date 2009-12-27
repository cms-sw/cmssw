#include "EventFilter/Utilities/interface/DebugUtils.h"

#include <sstream>
#include <iomanip>
#include <cstdio>

using namespace std;

string evf::dumpFrame(unsigned char* data, unsigned int len)
{
  
  ostringstream out;
  char left[40];
  char right[40];
      
  //  LOG4CPLUS_ERROR(adapter_->getApplicationLogger(),toolbox::toString("Byte  0  1  2  3  4  5  6  7\n"));
  out << "Byte:  0  1  2  3  4  5  6  7\n";
      
  int c = 0;
  int pos = 0;
      
      
  for (unsigned int i = 0; i < (len/8); i++) {
    int rpos = 0;
    c += 7;
    for (pos = 0; pos < 8*3; pos += 3) {
      sprintf (&left[pos],"%2.2x ", ((unsigned char*)data)[c]);
      sprintf (&right[rpos],"%1c", ((data[c] > 32) && (data[c] < 127)) ? data[c]: '.' );
      rpos += 1;
      c--;
    }
    c += 9;
    out << setw(4) << setfill('0') << c-8 << ": " << left << "  ||  " 
	<< right << endl;

  }	
  return out.str();
}   
