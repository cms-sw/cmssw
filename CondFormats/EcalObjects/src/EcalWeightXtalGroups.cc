/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"

EcalWeightXtalGroups::EcalWeightXtalGroups() {
}

EcalWeightXtalGroups::~EcalWeightXtalGroups() {

}

void
EcalWeightXtalGroups::setValue(const uint32_t& xtal, const EcalXtalGroupId& group) {
  map_[xtal] = group;
}

