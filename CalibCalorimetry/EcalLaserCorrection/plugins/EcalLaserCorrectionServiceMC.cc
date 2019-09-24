//
// Toyoko Orimoto (Caltech), 10 July 2007
// Andrea Massironi, 3 Aug 2019
//

// system include files
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbServiceMC.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecordMC.h"

#include "CalibCalorimetry/EcalLaserCorrection/plugins/EcalLaserCorrectionServiceMC.h"

EcalLaserCorrectionServiceMC::EcalLaserCorrectionServiceMC(const edm::ParameterSet& fConfig)
    : ESProducer()
{

  setWhatProduced(this);

}

EcalLaserCorrectionServiceMC::~EcalLaserCorrectionServiceMC() {
}

//
// member functions
//

// ------------ method called to produce the data  ------------
std::shared_ptr<EcalLaserDbServiceMC> EcalLaserCorrectionServiceMC::produce(const EcalLaserDbRecordMC& record) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  host->ifRecordChanges<EcalLinearCorrectionsRcd>(record,
                                                  [this, h = host.get()](auto const& rec) { setupLinear(rec, *h); });

  host->ifRecordChanges<EcalLaserAPDPNRatiosMCRcd>(record,
                                                   [this, h = host.get()](auto const& rec) { setupApdpn(rec, *h); });

  host->ifRecordChanges<EcalLaserAPDPNRatiosRefRcd>(
      record, [this, h = host.get()](auto const& rec) { setupApdpnRef(rec, *h); });

  host->ifRecordChanges<EcalLaserAlphasRcd>(record, [this, h = host.get()](auto const& rec) { setupAlpha(rec, *h); });

  return host;  // automatically converts to std::shared_ptr<EcalLaserDbServiceMC>
}

void EcalLaserCorrectionServiceMC::setupAlpha(const EcalLaserAlphasRcd& fRecord, EcalLaserDbServiceMC& service) {
  edm::ESHandle<EcalLaserAlphas> item;
  fRecord.get(item);
  service.setAlphaData(item.product());
}

void EcalLaserCorrectionServiceMC::setupApdpnRef(const EcalLaserAPDPNRatiosRefRcd& fRecord,
                                                 EcalLaserDbServiceMC& service) {
  edm::ESHandle<EcalLaserAPDPNRatiosRef> item;
  fRecord.get(item);
  service.setAPDPNRefData(item.product());
}

void EcalLaserCorrectionServiceMC::setupApdpn(const EcalLaserAPDPNRatiosMCRcd& fRecord, EcalLaserDbServiceMC& service) {
  edm::ESHandle<EcalLaserAPDPNRatiosMC> item;
  fRecord.get(item);
  service.setAPDPNData(item.product());
}

void EcalLaserCorrectionServiceMC::setupLinear(const EcalLinearCorrectionsRcd& fRecord, EcalLaserDbServiceMC& service) {
  edm::ESHandle<EcalLinearCorrections> item;
  fRecord.get(item);
  service.setLinearCorrectionsData(item.product());
}

DEFINE_FWK_EVENTSETUP_MODULE(EcalLaserCorrectionServiceMC);
