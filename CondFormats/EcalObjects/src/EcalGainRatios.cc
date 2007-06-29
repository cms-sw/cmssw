/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalGainRatios.cc,v 1.2 2006/02/23 16:56:35 rahatlou Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

EcalGainRatios::EcalGainRatios() {
}

EcalGainRatios::~EcalGainRatios() {

}

void
EcalGainRatios::setValue(const uint32_t& id, const EcalMGPAGainRatio & value) {
  map_[id] = value;
}

// safer than mutable vectors...
void EcalGainRatios::update() const {
  const_cast<EcalGainRatios&>(*this).doUpdate();
}
 
void EcalGainRatios::doUpdate(){
  m_hashedCont.load(getMap().begin(),getMap().end());
}
