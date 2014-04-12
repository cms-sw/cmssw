// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zNotIso() {

 
TCut kin_commonButIso("zGoldenDau1Pt> 20 && zGoldenDau2Pt>20 && (zGoldenDau1Iso03SumPt> 3.0 || zGoldenDau2Iso03SumPt >3.0) && abs(zGoldenDau1Eta)<2.4 &&  abs(zGoldenDau2Eta)<2.4  && (zGoldenDau1HLTBit==1 || zGoldenDau2HLTBit==1)  && abs(zGoldenDau1dxyFromBS)<0.2 && abs(zGoldenDau2dxyFromBS)<0.2 ");
 

 makePlots("zGoldenMass", "", kin_commonButIso + ( ( dau1Loose  && dau2TightWP1_hltAlso ) || ( dau2Loose  && dau1TightWP1_hltAlso )) , 5, "zNotIso",  0.001, 200, 0 ,200, true, true);




}


