#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTCrate.h"
#include <vector>
#include <iostream>
using std::vector;
using std::endl;
using std::cout;

int main(){
  L1RCTCrate crate(0);
  vector<vector<unsigned short> > b(7,vector<unsigned short>(64));
  vector<unsigned short> hf(8);
  crate.input(b,hf);
  crate.print();
}
