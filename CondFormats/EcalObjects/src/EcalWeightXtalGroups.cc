/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalWeightXtalGroups.cc,v 1.3 2007/06/29 07:04:31 innocent Exp $
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
 
void EcalWeightXtalGroups::doUpdate(){
  m_hashedCont.load(getMap().begin(),getMap().end());
}
