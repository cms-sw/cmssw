//
// Toyoko Orimoto (Caltech), 10 July 2007
// Andrea Massironi, 3 Aug 2019
//

// system include files
#include <iostream>
#include <fstream>
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecordMC.h"

#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosMCRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class EcalLaserCorrectionServiceMC : public edm::ESProducer {
public:
  EcalLaserCorrectionServiceMC(const edm::ParameterSet&);
  ~EcalLaserCorrectionServiceMC() override;

  std::shared_ptr<EcalLaserDbService> produce(const EcalLaserDbRecordMC&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using HostType = edm::ESProductHost<EcalLaserDbService,
                                      EcalLaserAlphasRcd,
                                      EcalLaserAPDPNRatiosRefRcd,
                                      EcalLaserAPDPNRatiosMCRcd,
                                      EcalLinearCorrectionsRcd>;

  // ----------member data ---------------------------
  edm::ReusableObjectHolder<HostType> holder_;

  edm::ESGetToken<EcalLaserAlphas, EcalLaserAlphasRcd> alphaToken_;
  edm::ESGetToken<EcalLaserAPDPNRatiosRef, EcalLaserAPDPNRatiosRefRcd> apdpnRefToken_;
  edm::ESGetToken<EcalLaserAPDPNRatios, EcalLaserAPDPNRatiosMCRcd> apdpnToken_;
  edm::ESGetToken<EcalLinearCorrections, EcalLinearCorrectionsRcd> linearToken_;
};

EcalLaserCorrectionServiceMC::EcalLaserCorrectionServiceMC(const edm::ParameterSet& fConfig) : ESProducer() {
  // the following line is needed to tell the framework what
  // data is being produced
  // setWhatProduced (this, (dependsOn (&EcalLaserCorrectionServiceMC::apdpnCallback)));

  auto cc = setWhatProduced(this);
  alphaToken_ = cc.consumes();
  apdpnRefToken_ = cc.consumes();
  apdpnToken_ = cc.consumes();
  linearToken_ = cc.consumes();

  //now do what ever other initialization is needed
}

EcalLaserCorrectionServiceMC::~EcalLaserCorrectionServiceMC() {}

//
// member functions
//
void EcalLaserCorrectionServiceMC::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("EcalLaserCorrectionServiceMC", desc);
}

// ------------ method called to produce the data  ------------
std::shared_ptr<EcalLaserDbService> EcalLaserCorrectionServiceMC::produce(const EcalLaserDbRecordMC& record) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  host->ifRecordChanges<EcalLinearCorrectionsRcd>(
      record, [this, h = host.get()](auto const& rec) { h->setLinearCorrectionsData(&rec.get(linearToken_)); });

  host->ifRecordChanges<EcalLaserAPDPNRatiosMCRcd>(
      record, [this, h = host.get()](auto const& rec) { h->setAPDPNData(&rec.get(apdpnToken_)); });

  host->ifRecordChanges<EcalLaserAPDPNRatiosRefRcd>(
      record, [this, h = host.get()](auto const& rec) { h->setAPDPNRefData(&rec.get(apdpnRefToken_)); });

  host->ifRecordChanges<EcalLaserAlphasRcd>(
      record, [this, h = host.get()](auto const& rec) { h->setAlphaData(&rec.get(alphaToken_)); });

  return host;  // automatically converts to std::shared_ptr<EcalLaserDbService>
}

DEFINE_FWK_EVENTSETUP_MODULE(EcalLaserCorrectionServiceMC);
