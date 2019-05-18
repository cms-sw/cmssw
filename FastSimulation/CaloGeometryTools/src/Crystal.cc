//FAMOS headers
#include "FastSimulation/CaloGeometryTools/interface/Crystal.h"

Crystal::Crystal(const DetId& cell, const BaseCrystal* xtal) : cellid_(cell), myCrystal_(xtal) {
  neighbours_.resize(8);
}
