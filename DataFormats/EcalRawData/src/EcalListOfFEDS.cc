#include "DataFormats/EcalRawData/interface/EcalListOfFEDS.h"
#include <iostream>

EcalListOfFEDS::EcalListOfFEDS() { list_of_feds.clear(); }

void EcalListOfFEDS::AddFED(int fed) { list_of_feds.push_back(fed); }
