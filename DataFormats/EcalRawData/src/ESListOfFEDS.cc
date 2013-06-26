#include <iostream>
#include "DataFormats/EcalRawData/interface/ESListOfFEDS.h"

ESListOfFEDS::ESListOfFEDS() {
 list_of_feds.clear();
}

void ESListOfFEDS::AddFED(int fed) {
 list_of_feds.push_back(fed);
}


