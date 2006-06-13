#include <iostream>
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTElectronIsolationCard.h"

int main() {
  L1RCTElectronIsolationCard eic(0,0);
  L1RCTRegion r0;
  L1RCTRegion r1;
  //This should report 1 isolated electron of 100 
  r0.setEtIn7Bits(0,0,100);
  eic.setRegion(0,&r0);
  eic.setRegion(1,&r1);

  eic.fillElectronCandidates();
  eic.print();

  //This should report *no* electrons
  r0.setHE_FGBit(0,0,1);
  eic.fillElectronCandidates();
  eic.print();

  //This should report only a nonisolated electron of 100
  r0.setHE_FGBit(0,0,0);
  r0.setHE_FGBit(0,1,1);
  eic.fillElectronCandidates();
  eic.print();

  //This should report an isolated electron of 80 and a nonisolated of 100
  r0.setEtIn7Bits(2,0,80);
  eic.fillElectronCandidates();
  eic.print();
			     

}
