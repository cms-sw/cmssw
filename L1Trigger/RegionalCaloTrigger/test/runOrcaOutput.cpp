#include <iostream>
#include <fstream>
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCT.h"

int main(){
  char inputname[256];
  std::string filename("../data/TPGcalc.txt");
  L1RCT rct(filename);
  for(int i=1; i<=100;i++){
    sprintf(inputname,"data/rct-input-%i.dat",i);
    rct.fileInput(inputname);
    rct.processEvent();
    rct.printJSC();
  }
}
