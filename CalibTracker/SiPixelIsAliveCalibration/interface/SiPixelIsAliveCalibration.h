
#include "CalibFormats/SiPixelObjects/interface/PixelCalib.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "TObjArray.h"

// system include files
#include <memory>

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"

#include "CalibFormats/SiPixelObjects/interface/PixelCalib.h"
#include "CondTools/SiPixel/interface/SiPixelGainCalibrationService.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include <iostream>
#include <map>

class SiPixelIsAliveCalibration : public edm::EDAnalyzer {
   public:
      explicit SiPixelIsAliveCalibration(const edm::ParameterSet&);
      ~SiPixelIsAliveCalibration();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      bool use_calib_;
      std::string src_;
      unsigned int eventno_counter;
      std::string inputconfigfile_;
      void fill(unsigned int detid, unsigned int adc, unsigned int icol, unsigned int irow, unsigned int ntimes);
      void init(unsigned detid);
      TObjArray *theHistos_;
      PixelCalib *calib_;
      TString makeHistName(unsigned int detid);
      edm::Service<TFileService> therootfileservice_;
      edm::ESHandle<TrackerGeometry> geom_;

      void writeOutRootMacro();
      // ----------member data ---------------------------
};
