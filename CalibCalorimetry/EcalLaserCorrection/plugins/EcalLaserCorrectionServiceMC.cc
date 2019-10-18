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

class EcalLaserCorrectionServiceMC : public edm::ESProducer {
public:
  EcalLaserCorrectionServiceMC(const edm::ParameterSet&);
  ~EcalLaserCorrectionServiceMC() override;

  std::shared_ptr<EcalLaserDbService> produce(const EcalLaserDbRecordMC&);

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

  //  std::vector<std::string> mDumpRequest;
  //  std::ostream* mDumpStream;
};

EcalLaserCorrectionServiceMC::EcalLaserCorrectionServiceMC(const edm::ParameterSet& fConfig)
    : ESProducer()
//    mDumpRequest (),
//    mDumpStream(0)
{
  //the following line is needed to tell the framework what
  // data is being produced
  //  setWhatProduced (this, (dependsOn (&EcalLaserCorrectionServiceMC::apdpnCallback)));

  setWhatProduced(this)
      .setConsumes(alphaToken_)
      .setConsumes(apdpnRefToken_)
      .setConsumes(apdpnToken_)
      .setConsumes(linearToken_);

  //now do what ever other initialization is needed

  //  mDumpRequest = fConfig.getUntrackedParameter <std::vector <std::string> > ("dump", std::vector<std::string>());
  //  if (!mDumpRequest.empty()) {
  //    std::string otputFile = fConfig.getUntrackedParameter <std::string> ("file", "");
  //    mDumpStream = otputFile.empty () ? &std::cout : new std::ofstream (otputFile.c_str());
  //  }
}

EcalLaserCorrectionServiceMC::~EcalLaserCorrectionServiceMC() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //  if (mDumpStream != &std::cout) delete mDumpStream;
}

//
// member functions
//

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
