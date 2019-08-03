//
// Toyoko Orimoto (Caltech), 10 July 2007
//

#ifndef EcalLaserDbServiceMC_h
#define EcalLaserDbServiceMC_h

#include <memory>
#include <tbb/concurrent_unordered_set.h>


#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosMC.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"


class EcalLaserDbServiceMC {
 public:
  EcalLaserDbServiceMC ();

  const EcalLaserAlphas* getAlphas () const;
  const EcalLaserAPDPNRatiosRef* getAPDPNRatiosRef () const;
  const EcalLaserAPDPNRatiosMC* getAPDPNRatiosMC () const;
  const EcalLinearCorrections* getLinearCorrections () const;
  float getLaserCorrection (DetId const & xid, edm::Timestamp const & iTime) const;

  void setAlphaData (const EcalLaserAlphas* fItem) {mAlphas_ = fItem;}
  void setAPDPNRefData (const EcalLaserAPDPNRatiosRef* fItem) {mAPDPNRatiosRef_ = fItem;}
  void setAPDPNData (const EcalLaserAPDPNRatiosMC* fItem) {mAPDPNRatiosMC_ = fItem;}
  void setLinearCorrectionsData (const EcalLinearCorrections* fItem) {mLinearCorrections_ = fItem;}

 private:

  const EcalLaserAlphas* mAlphas_;
  const EcalLaserAPDPNRatiosRef* mAPDPNRatiosRef_;
  const EcalLaserAPDPNRatiosMC* mAPDPNRatiosMC_;  
  const EcalLinearCorrections* mLinearCorrections_;  

  typedef tbb::concurrent_unordered_set<uint32_t> ErrorMapT;
  mutable ErrorMapT channelsWithInvalidCorrection_;

};

#endif
