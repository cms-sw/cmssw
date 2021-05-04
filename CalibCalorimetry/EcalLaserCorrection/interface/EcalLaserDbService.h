//
// Toyoko Orimoto (Caltech), 10 July 2007
// Fabrice Couderc (Saclay), 16 March 2020 : add protection if t > t3 +deltat : t = t3 + deltat
//

#ifndef EcalLaserDbService_h
#define EcalLaserDbService_h

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
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"

class EcalLaserDbService {
public:
  EcalLaserDbService();

  const EcalLaserAlphas* getAlphas() const;
  const EcalLaserAPDPNRatiosRef* getAPDPNRatiosRef() const;
  const EcalLaserAPDPNRatios* getAPDPNRatios() const;
  const EcalLinearCorrections* getLinearCorrections() const;
  float getLaserCorrection(DetId const& xid, edm::Timestamp const& iTime) const;

  void setAlphaData(const EcalLaserAlphas* fItem) { mAlphas_ = fItem; }
  void setAPDPNRefData(const EcalLaserAPDPNRatiosRef* fItem) { mAPDPNRatiosRef_ = fItem; }
  void setAPDPNData(const EcalLaserAPDPNRatios* fItem) { mAPDPNRatios_ = fItem; }
  void setLinearCorrectionsData(const EcalLinearCorrections* fItem) { mLinearCorrections_ = fItem; }

  void setMaxExtrapolationTimeInSec(int deltaT) { maxExtrapolationTime_ = ((unsigned long long)deltaT) << 32; }

private:
  const EcalLaserAlphas* mAlphas_;
  const EcalLaserAPDPNRatiosRef* mAPDPNRatiosRef_;
  const EcalLaserAPDPNRatios* mAPDPNRatios_;
  const EcalLinearCorrections* mLinearCorrections_;
  unsigned long long maxExtrapolationTime_;

  typedef tbb::concurrent_unordered_set<uint32_t> ErrorMapT;
  mutable ErrorMapT channelsWithInvalidCorrection_;
};

#endif
