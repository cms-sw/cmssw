#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"
#include <iostream>

void DTGeomSupplier::print(const DTTrigData* trig) const {
  trig->print();
  std::cout << " Local (pos)(dir): " << localPosition(trig)
            << localDirection(trig) << std::endl;
  std::cout << " CMS (pos)(dir): " << CMSPosition(trig) 
            << CMSDirection(trig) << std::endl;
}    

