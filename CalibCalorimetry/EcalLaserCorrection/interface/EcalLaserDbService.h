//
// Toyoko Orimoto (Caltech), 10 July 2007
//

#ifndef EcalLaserDbService_h
#define EcalLaserDbService_h

#include <memory>
#include <map>

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

//#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
//#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
//#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

class EcalLaserAlphas;
class EcalLaserAPDPNRatiosRef;
class EcalLaserAPDPNRatios;


class EcalLaserDbService {
 public:
  EcalLaserDbService ();
  EcalLaserDbService (const edm::ParameterSet&);

  //  const EcalLaserAlphas* getAlpha (const EBDetId& fId) const;
  //  const EcalLaserAPDPNRatiosRef* getAPDPNRatiosRef (const EBDetId& fId) const;
  //  const EcalLaserAPDPNRatios* getAPDPNRatio (const EBDetId& fId) const;

  const EcalLaserAlphas* getAlphas () const;
  const EcalLaserAPDPNRatiosRef* getAPDPNRatiosRef () const;
  const EcalLaserAPDPNRatios* getAPDPNRatios () const;

  //  const HcalQIEShape* getHcalShape () const;
  //  const HcalElectronicsMap* getHcalMapping () const;
  
  void setData (const EcalLaserAlphas* fItem) {mAlphas = fItem;}
  void setData (const EcalLaserAPDPNRatiosRef* fItem) {mAPDPNRatiosRef = fItem;}
  void setData (const EcalLaserAPDPNRatios* fItem) {mAPDPNRatios = fItem;}

  //  void setData (const HcalElectronicsMap* fItem) {mElectronicsMap = fItem;}
 private:
  const EcalLaserAlphas* mAlphas;
  const EcalLaserAPDPNRatiosRef* mAPDPNRatiosRef;
  const EcalLaserAPDPNRatios* mAPDPNRatios;

  //  const HcalElectronicsMap* mElectronicsMap;
};

#endif
