#ifndef CondTools_SiPixel_SiPixelTemplateDBObjectReader_h
#define CondTools_SiPixel_SiPixelTemplateDBObjectReader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObject0TRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObject38TRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObject4TRcd.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"


class SiPixelTemplateDBObjectReader : public edm::EDAnalyzer {
   public:
      explicit SiPixelTemplateDBObjectReader(const edm::ParameterSet&);
      ~SiPixelTemplateDBObjectReader();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
		
			edm::ESWatcher<SiPixelTemplateDBObjectESProducerRcd>  SiPixTemplDBObjectWatcher_;
      edm::ESWatcher<SiPixelTemplateDBObject0TRcd>  SiPixTemplDBObj0TWatcher_;
			edm::ESWatcher<SiPixelTemplateDBObject38TRcd> SiPixTemplDBObj38TWatcher_;
			edm::ESWatcher<SiPixelTemplateDBObject4TRcd>  SiPixTemplDBObj4TWatcher_;
			
      std::string theTemplateCalibrationLocation;
      bool theDetailedTemplateDBErrorOutput;
      bool theFullTemplateDBOutput;
			float theMagField;
			bool testStandalone;
			
			SiPixelTemplateDBObject dbobject;
			bool hasTriggeredWatcher;			
			

};

#endif
