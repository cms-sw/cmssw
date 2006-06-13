#include "L1RCT.h"

int main(){
  L1RCT rct;
  rct.randomInput();
  //rct.print();
  rct.processEvent();
  rct.printJSC();
}
