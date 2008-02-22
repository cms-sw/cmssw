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

  double getTPGInGeV(const edm::EventSetup & evtSetup, const EcalTriggerPrimitiveDigi & tpDigi) ;
  double getTPGInGeV(const edm::EventSetup & evtSetup, int ADC, const EcalTrigTowerDetId & towerId) ;

  int    getLinearizedTPG(const edm::EventSetup & evtSetup, int ADC, const EcalTrigTowerDetId & towerId) ;
  int    getTPGInADC(const edm::EventSetup & evtSetup, double energy, const EcalTrigTowerDetId & towerId) ;

 private:
};

#endif
