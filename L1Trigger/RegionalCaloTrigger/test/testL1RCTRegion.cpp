#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTRegion.h"
#include <vector>
using std::vector;

int main(){
  L1RCTRegion region;
  region.print();
  region.setEtIn7Bits(0,0,10);
  region.print();
  region.setEtIn9Bits(0,0,10);
  region.print();
}
  
  
