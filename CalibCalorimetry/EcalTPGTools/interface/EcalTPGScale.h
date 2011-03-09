#ifndef ECALTPGSCALE_H
#define ECALTPGSCALE_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

class EcalTPGScale
{
 public:
  EcalTPGScale() ;
  ~EcalTPGScale() ;

  void setEventSetup(const edm::EventSetup & evtSetup) ;

  double getTPGInGeV(const EcalTriggerPrimitiveDigi & tpDigi) ;
  double getTPGInGeV(uint ADC, const EcalTrigTowerDetId & towerId) ;

  uint    getLinearizedTPG(uint ADC, const EcalTrigTowerDetId & towerId) ;
  uint    getTPGInADC(double energy, const EcalTrigTowerDetId & towerId) ;

 private:
  const edm::EventSetup * setup_ ;

};

#endif
