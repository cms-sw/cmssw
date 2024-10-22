//
// Toyoko Orimoto (Caltech), 10 July 2007
// Fabrice Couderc, 16 March 2020 (add protection for extrapolation if t > t3 + delta t : t = t3 + delta t
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
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"

class EcalLaserCorrectionService : public edm::ESProducer {
public:
  EcalLaserCorrectionService(const edm::ParameterSet&);
  ~EcalLaserCorrectionService() override;

  std::shared_ptr<EcalLaserDbService> produce(const EcalLaserDbRecord&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  using HostType = edm::ESProductHost<EcalLaserDbService,
                                      EcalLaserAlphasRcd,
                                      EcalLaserAPDPNRatiosRefRcd,
                                      EcalLaserAPDPNRatiosRcd,
                                      EcalLinearCorrectionsRcd>;

  // ----------member data ---------------------------
  edm::ReusableObjectHolder<HostType> holder_;

  edm::ESGetToken<EcalLaserAlphas, EcalLaserAlphasRcd> alphaToken_;
  edm::ESGetToken<EcalLaserAPDPNRatiosRef, EcalLaserAPDPNRatiosRefRcd> apdpnRefToken_;
  edm::ESGetToken<EcalLaserAPDPNRatios, EcalLaserAPDPNRatiosRcd> apdpnToken_;
  edm::ESGetToken<EcalLinearCorrections, EcalLinearCorrectionsRcd> linearToken_;

  int maxExtrapolationTimeInSec_;

  //  std::vector<std::string> mDumpRequest;
  //  std::ostream* mDumpStream;
};

EcalLaserCorrectionService::EcalLaserCorrectionService(const edm::ParameterSet& fConfig)
    : ESProducer()
//    mDumpRequest (),
//    mDumpStream(0)
{
  //the following line is needed to tell the framework what
  // data is being produced
  //  setWhatProduced (this, (dependsOn (&EcalLaserCorrectionService::apdpnCallback)));

  auto cc = setWhatProduced(this);
  alphaToken_ = cc.consumes();
  apdpnRefToken_ = cc.consumes();
  apdpnToken_ = cc.consumes();
  linearToken_ = cc.consumes();

  maxExtrapolationTimeInSec_ = fConfig.getParameter<unsigned int>("maxExtrapolationTimeInSec");

  //now do what ever other initialization is needed

  //  mDumpRequest = fConfig.getUntrackedParameter <std::vector <std::string> > ("dump", std::vector<std::string>());
  //  if (!mDumpRequest.empty()) {
  //    std::string otputFile = fConfig.getUntrackedParameter <std::string> ("file", "");
  //    mDumpStream = otputFile.empty () ? &std::cout : new std::ofstream (otputFile.c_str());
  //  }
}

EcalLaserCorrectionService::~EcalLaserCorrectionService() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //  if (mDumpStream != &std::cout) delete mDumpStream;
}

//
// member functions
//

// ------------ method called to produce the data  ------------
std::shared_ptr<EcalLaserDbService> EcalLaserCorrectionService::produce(const EcalLaserDbRecord& record) {
  auto host = holder_.makeOrGet([]() { return new HostType; });

  host.get()->setMaxExtrapolationTimeInSec(maxExtrapolationTimeInSec_);

  host->ifRecordChanges<EcalLinearCorrectionsRcd>(
      record, [this, h = host.get()](auto const& rec) { h->setLinearCorrectionsData(&rec.get(linearToken_)); });

  host->ifRecordChanges<EcalLaserAPDPNRatiosRcd>(
      record, [this, h = host.get()](auto const& rec) { h->setAPDPNData(&rec.get(apdpnToken_)); });

  host->ifRecordChanges<EcalLaserAPDPNRatiosRefRcd>(
      record, [this, h = host.get()](auto const& rec) { h->setAPDPNRefData(&rec.get(apdpnRefToken_)); });

  host->ifRecordChanges<EcalLaserAlphasRcd>(
      record, [this, h = host.get()](auto const& rec) { h->setAlphaData(&rec.get(alphaToken_)); });

  return host;  // automatically converts to std::shared_ptr<EcalLaserDbService>
}

void EcalLaserCorrectionService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<unsigned int>("maxExtrapolationTimeInSec", 0);
  descriptions.add("EcalLaserCorrectionService", desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(EcalLaserCorrectionService);
