//
// Toyoko Orimoto (Caltech), 10 July 2007
//

#ifndef EcalLaserDbService_h
#define EcalLaserDbService_h

#include <memory>
#include <map>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

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

  const EcalLaserAlphas* getAlphas () const;
  const EcalLaserAPDPNRatiosRef* getAPDPNRatiosRef () const;
  const EcalLaserAPDPNRatios* getAPDPNRatios () const;
  float getLaserCorrection (DetId const & xid, edm::Timestamp const & iTime) const;

  void setData (const EcalLaserAlphas* fItem) {mAlphas = fItem;}
  void setData (const EcalLaserAPDPNRatiosRef* fItem) {mAPDPNRatiosRef = fItem;}
  void setData (const EcalLaserAPDPNRatios* fItem) {mAPDPNRatios = fItem;}
  //  void setVerbosity (const bool verb) const {verbose = verb;}

 private:
  const EcalLaserAlphas* mAlphas;
  const EcalLaserAPDPNRatiosRef* mAPDPNRatiosRef;
  const EcalLaserAPDPNRatios* mAPDPNRatios;

  //  bool verbose;

};

#endif
