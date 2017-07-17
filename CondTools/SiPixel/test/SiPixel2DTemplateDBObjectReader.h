#ifndef CondTools_SiPixel_SiPixel2DTemplateDBObjectReader_h
#define CondTools_SiPixel_SiPixel2DTemplateDBObjectReader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixel2DTemplateDBObjectRcd.h"
#include "CalibTracker/Records/interface/SiPixel2DTemplateDBObjectESProducerRcd.h"


class SiPixel2DTemplateDBObjectReader : public edm::EDAnalyzer {
   public:
      explicit SiPixel2DTemplateDBObjectReader(const edm::ParameterSet&);
      ~SiPixel2DTemplateDBObjectReader();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
		
			edm::ESWatcher<SiPixel2DTemplateDBObjectESProducerRcd>  SiPix2DTemplDBObjectWatcher_;
      edm::ESWatcher<SiPixel2DTemplateDBObjectRcd>  SiPix2DTemplDBObjWatcher_;
			
      std::string the2DTemplateCalibrationLocation;
      bool theDetailed2DTemplateDBErrorOutput;
      bool theFull2DTemplateDBOutput;
			bool testGlobalTag;
			
			SiPixel2DTemplateDBObject dbobject;
			bool hasTriggeredWatcher;			
			

};

#endif
