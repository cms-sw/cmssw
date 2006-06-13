#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

int main(){
  L1RCT rct;
  rct.randomInput();
  //rct.print();
  rct.processEvent();
  rct.printJSC();
}
