#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLookupTables.h"
#include <iostream>
using std::cout;
using std::endl;
int main() {
  std::string filename("../data/TPGcalc.txt");
  L1RCTLookupTables lut(filename);
  cout << lut.lookup(0,0,0,0,0,0) << " should equal 0" << endl;
  cout << lut.lookup(2,0,0,0,0,0) << " should equal 514" << endl;
  cout << lut.lookup(10,0,0,0,0,0) << " should equal 133642 " << endl;
  cout << lut.lookup(10,0,1,0,0,0) << " should equal 133770 " << endl;
  cout << lut.lookup(0,10,0,0,0,0) << " should equal 133770 " << endl;
  cout << lut.lookup(0,10,1,0,0,0) << " should equal 133770 " << endl;
  cout << lut.lookup(255,0,0,0,0,0) << " should equal 196479 " << endl;
}
  
