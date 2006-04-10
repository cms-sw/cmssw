#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include <iostream>
using std::cout;
using std::endl;
int main() {
  L1RCTLookupTables lut;
  cout << lut.lookup(0,0,0) << " should equal 0" << endl;
  cout << lut.lookup(2,0,0) << " should equal 258" << endl;
  cout << lut.lookup(10,0,0) << " should equal 132362 " << endl;
  cout << lut.lookup(10,0,1) << " should equal 197898 " << endl;
  cout << lut.lookup(0,10,0) << " should equal 197898 " << endl;
  cout << lut.lookup(0,10,1) << " should equal 197898 " << endl;
  cout << lut.lookup(255,0,0) << " should equal 163839 " << endl;
}
  
