#ifndef ECALTPGSCALE_H
#define ECALTPGSCALE_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveDigi.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGLutGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"

class EcalTPGScale {
public:
  struct Tokens {
    Tokens(edm::ConsumesCollector);
    const edm::ESGetToken<EcalTPGPhysicsConst, EcalTPGPhysicsConstRcd> physConstToken_;
    const edm::ESGetToken<EcalTPGLutGroup, EcalTPGLutGroupRcd> lutGrpToken_;
    const edm::ESGetToken<EcalTPGLutIdMap, EcalTPGLutIdMapRcd> lutMapToken_;
  };

  EcalTPGScale(Tokens const&, const edm::EventSetup& evtSetup);

  double getTPGInGeV(const EcalTriggerPrimitiveDigi& tpDigi) const;
  double getTPGInGeV(unsigned int ADC, const EcalTrigTowerDetId& towerId) const;

  unsigned int getLinearizedTPG(unsigned int ADC, const EcalTrigTowerDetId& towerId) const;
  unsigned int getTPGInADC(double energy, const EcalTrigTowerDetId& towerId) const;

private:
  EcalTPGPhysicsConst const& phys_;
  EcalTPGLutGroup const& lutGrp_;
  EcalTPGLutIdMap const& lut_;
};

#endif
