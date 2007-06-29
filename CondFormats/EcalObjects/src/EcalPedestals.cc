#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/EcalDetId/interface/EcalContainer.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include <algorithm>

EcalPedestals::Item::Zero EcalPedestals::Item::zero;

EcalPedestals::EcalPedestals(){}
EcalPedestals::~EcalPedestals(){}


// safer than mutable vectors...
void EcalPedestals::update() const {
  const_cast<EcalPedestals&>(*this).doUpdate();
}
 
void EcalPedestals::doUpdate(){
  m_hashedCont.load(m_pedestals.begin(),m_pedestals.end());
}
