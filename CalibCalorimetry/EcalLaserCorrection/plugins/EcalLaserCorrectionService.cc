//
// Toyoko Orimoto (Caltech), 10 July 2007
//

// system include files
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "CalibCalorimetry/EcalLaserCorrection/plugins/EcalLaserCorrectionService.h"


EcalLaserCorrectionService::EcalLaserCorrectionService( const edm::ParameterSet& fConfig)
  : ESProducer()
    //    mDumpRequest (),
    //    mDumpStream(0)
{
  //the following line is needed to tell the framework what
  // data is being produced
  //  setWhatProduced (this, (dependsOn (&EcalLaserCorrectionService::apdpnCallback)));

  setWhatProduced (this);

  //now do what ever other initialization is needed

  //  mDumpRequest = fConfig.getUntrackedParameter <std::vector <std::string> > ("dump", std::vector<std::string>());
  //  if (!mDumpRequest.empty()) {
  //    std::string otputFile = fConfig.getUntrackedParameter <std::string> ("file", "");
  //    mDumpStream = otputFile.empty () ? &std::cout : new std::ofstream (otputFile.c_str());
  //  }
}


EcalLaserCorrectionService::~EcalLaserCorrectionService()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  //  if (mDumpStream != &std::cout) delete mDumpStream;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
std::shared_ptr<EcalLaserDbService> EcalLaserCorrectionService::produce( const EcalLaserDbRecord& record)
{
  auto host = holder_.makeOrGet([]() {
    return new HostType;
  });

  host->ifRecordChanges<EcalLinearCorrectionsRcd>(record,
                                                  [this,h=host.get()](auto const& rec) {
    setupLinear(rec, *h);
  });

  host->ifRecordChanges<EcalLaserAPDPNRatiosRcd>(record,
                                                 [this,h=host.get()](auto const& rec) {
    setupApdpn(rec, *h);
  });

  host->ifRecordChanges<EcalLaserAPDPNRatiosRefRcd>(record,
                                                    [this,h=host.get()](auto const& rec) {
    setupApdpnRef(rec, *h);
  });

  host->ifRecordChanges<EcalLaserAlphasRcd>(record,
                                            [this,h=host.get()](auto const& rec) {
    setupAlpha(rec, *h);
  });

  return host; // automatically converts to std::shared_ptr<EcalLaserDbService>
}

void EcalLaserCorrectionService::setupAlpha(const EcalLaserAlphasRcd& fRecord,
                                            EcalLaserDbService& service) {
  edm::ESHandle <EcalLaserAlphas> item;
  fRecord.get (item);
  service.setAlphaData (item.product ());
}

void EcalLaserCorrectionService::setupApdpnRef(const EcalLaserAPDPNRatiosRefRcd& fRecord,
                                               EcalLaserDbService& service) {
  edm::ESHandle <EcalLaserAPDPNRatiosRef> item;
  fRecord.get (item);
  service.setAPDPNRefData (item.product ());
}

void EcalLaserCorrectionService::setupApdpn(const EcalLaserAPDPNRatiosRcd& fRecord,
                                            EcalLaserDbService& service) {
  edm::ESHandle <EcalLaserAPDPNRatios> item;
  fRecord.get (item);
  service.setAPDPNData (item.product ());
}

void EcalLaserCorrectionService::setupLinear(const EcalLinearCorrectionsRcd& fRecord,
                                             EcalLaserDbService& service) {
  edm::ESHandle <EcalLinearCorrections> item;
  fRecord.get (item);
  service.setLinearCorrectionsData (item.product ());
}
