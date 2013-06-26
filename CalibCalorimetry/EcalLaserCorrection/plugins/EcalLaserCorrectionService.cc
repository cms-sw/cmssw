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
  : ESProducer(),
    mService_ ( new EcalLaserDbService ())
    //    mDumpRequest (),
    //    mDumpStream(0)
{
  //the following line is needed to tell the framework what
  // data is being produced
  //  setWhatProduced (this, (dependsOn (&EcalLaserCorrectionService::apdpnCallback)));

  setWhatProduced (this, (dependsOn (&EcalLaserCorrectionService::alphaCallback) &
     			  (&EcalLaserCorrectionService::apdpnRefCallback) &
     			  (&EcalLaserCorrectionService::apdpnCallback) &
                          (&EcalLaserCorrectionService::linearCallback)
                          )
     		   );

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
boost::shared_ptr<EcalLaserDbService> EcalLaserCorrectionService::produce( const EcalLaserDbRecord& )
{
  return mService_;
}

void EcalLaserCorrectionService::alphaCallback (const EcalLaserAlphasRcd& fRecord) {
  edm::ESHandle <EcalLaserAlphas> item;
  fRecord.get (item);
  mService_->setAlphaData (item.product ());
}

void EcalLaserCorrectionService::apdpnRefCallback (const EcalLaserAPDPNRatiosRefRcd& fRecord) {
  edm::ESHandle <EcalLaserAPDPNRatiosRef> item;
  fRecord.get (item);
  mService_->setAPDPNRefData (item.product ());
}

void EcalLaserCorrectionService::apdpnCallback (const EcalLaserAPDPNRatiosRcd& fRecord) {
  edm::ESHandle <EcalLaserAPDPNRatios> item;
  fRecord.get (item);
  mService_->setAPDPNData (item.product ());
}

void EcalLaserCorrectionService::linearCallback (const EcalLinearCorrectionsRcd& fRecord) {
  edm::ESHandle <EcalLinearCorrections> item;
  fRecord.get (item);
  mService_->setLinearCorrectionsData (item.product ());
}
