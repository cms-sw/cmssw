// system include files
#include <memory>
#include <sys/stat.h>
#include <map>
#include <utility>
#include <vector>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondTools/SiPixel/interface/SiPixelGainCalibrationOfflineService.h"


#include "TH2F.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TKey.h"
#include "TString.h"
#include "TList.h"



class SiPixelGainCalibrationRejectNoisyAndDead : public edm::EDAnalyzer {
   public:
      explicit SiPixelGainCalibrationRejectNoisyAndDead(const edm::ParameterSet&);
      ~SiPixelGainCalibrationRejectNoisyAndDead();

   private:
      edm::ParameterSet conf_;
      SiPixelGainCalibrationOfflineService SiPixelGainCalibrationService_;
      SiPixelGainCalibrationOffline *theGainCalibrationDbInputOffline_;

      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      std::map < int,std::vector < std::pair < int,int > > > noisypixelkeeper;
      
      void fillDatabase(const edm::EventSetup&);
      void getNoisyPixels();
      void getDeadPixels();

      // ----------member data ---------------------------
      
      
      std::string noisypixellist_;
      bool insertnoisypixelsindb_;
      float pedlow_;
      float pedhi_;
      float gainlow_;
      float gainhi_;

};
