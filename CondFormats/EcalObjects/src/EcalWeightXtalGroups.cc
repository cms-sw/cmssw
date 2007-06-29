/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalWeightXtalGroups.cc,v 1.2 2006/02/23 16:56:35 rahatlou Exp $
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

// safer than mutable vectors...
void EcalWeightXtalGroups::update() const {
  const_cast<EcalWeightXtalGroups&>(*this).doUpdate();
}
 
voidEcalWeightXtalGroups::doUpdate(){
  m_hashedCont.load(getMap().begin(),getMap().end());
}
