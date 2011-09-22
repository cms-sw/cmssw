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
  double getTPGInGeV(unsigned int ADC, const EcalTrigTowerDetId & towerId) ;

  unsigned int    getLinearizedTPG(unsigned int ADC, const EcalTrigTowerDetId & towerId) ;
  unsigned int    getTPGInADC(double energy, const EcalTrigTowerDetId & towerId) ;

 private:
  const edm::EventSetup * setup_ ;

};

#endif
