#include <iostream>
#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"

EcalListOfFEDS::EcalListOfFEDS() {
 list_of_feds.clear();
}

void EcalListOfFEDS::AddFED(int fed) {
 list_of_feds.push_back(fed);
}


