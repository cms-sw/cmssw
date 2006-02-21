#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"

EcalWeightXtalGroups::EcalWeightXtalGroups() {
}

EcalWeightXtalGroups::~EcalWeightXtalGroups() {

}

void
EcalWeightXtalGroups::setValue(const uint32_t& xtal, const EcalXtalGroupId& group) {
  map_[xtal] = group;
}

