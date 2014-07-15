#ifndef CondTools_SiPixel_SiPixelGenErrorDBObjectReader_h
#define CondTools_SiPixel_SiPixelGenErrorDBObjectReader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "CalibTracker/Records/interface/SiPixelGenErrorDBObjectESProducerRcd.h"


class SiPixelGenErrorDBObjectReader : public edm::EDAnalyzer {
   public:
      explicit SiPixelGenErrorDBObjectReader(const edm::ParameterSet&);
      ~SiPixelGenErrorDBObjectReader();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
		
			edm::ESWatcher<SiPixelGenErrorDBObjectESProducerRcd>  SiPixGenerDBObjectWatcher_;
      edm::ESWatcher<SiPixelGenErrorDBObjectRcd>  SiPixGenerDBObjWatcher_;
			
      std::string theGenErrorCalibrationLocation;
      bool theDetailedGenErrorDBErrorOutput;
      bool theFullGenErrorDBOutput;
			bool testGlobalTag;
			
			SiPixelGenErrorDBObject dbobject;
			bool hasTriggeredWatcher;			
			

};

#endif
