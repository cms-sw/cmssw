#ifndef L1RCTLookupTables_h
#define L1RCTLookupTables_h

#include <math.h>
#include <iostream>
using std::cout;
using std::endl;

class L1RCTLookupTables {
 
 public:
  //function needs to output an unsigned long

  unsigned long lookup(unsigned short ecal,unsigned short hcal,
		       unsigned short fgbit);
  unsigned short lookup(unsigned short hfenergy);

 private:
  float convertEcal(unsigned short ecal);
  float convertHcal(unsigned short hcal);
  unsigned short calcHEBit(float ecal,float hcal);
  unsigned short calcActivityBit(float ecal,float hcal);
  unsigned long convertTo7Bits(float et);
  unsigned long convertTo9Bits(float et);
  unsigned long convertTo10Bits(float et);
};
#endif
