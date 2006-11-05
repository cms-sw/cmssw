#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

int main(){
  std::string filename("../data/TPGcalc.txt");
  L1RCT rct(filename);
  rct.randomInput();
  //rct.print();
  rct.processEvent();
  rct.printJSC();
}
