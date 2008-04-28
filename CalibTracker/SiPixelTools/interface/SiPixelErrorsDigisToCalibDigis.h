#ifndef SiPixelErrorsDigisToCalibDigis_h
#define SiPixelErrorsDigisToCalibDigis_h

/**  class SiPixelErrorsDigisToCalibDigis

Description: Create monitorElements for the Errors in created in the reduction of digis to calibDigis

**/

// Original Author: Ricardo Vasquez Sierra on April 9, 2008


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigi.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelCalibDigiError.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//
// class declaration
//

class SiPixelErrorsDigisToCalibDigis : public edm::EDAnalyzer {
   public:
      explicit SiPixelErrorsDigisToCalibDigis(const edm::ParameterSet&);
      ~SiPixelErrorsDigisToCalibDigis();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

      // store the number of error per detector id encountered
      // store the location of the mismatching error in the detector id

      edm::InputTag siPixelProducerLabel_;
      DQMStore* daqBE_;
      std::string outputFilename_;
      bool createOutputFile_;

      
      std::map<uint32_t, MonitorElement*> SiPixelErrorsDigisToCalibDigis_2DErrorInformation_; 
      
};

#endif
