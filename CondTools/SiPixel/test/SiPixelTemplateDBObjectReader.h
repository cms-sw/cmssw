#ifndef CondTools_SiPixel_SiPixelTemplateDBObjectReader_h
#define CondTools_SiPixel_SiPixelTemplateDBObjectReader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"


class SiPixelTemplateDBObjectReader : public edm::EDAnalyzer {
   public:
      explicit SiPixelTemplateDBObjectReader(const edm::ParameterSet&);
      ~SiPixelTemplateDBObjectReader();

			//		typedef std::vector<std::string> vstring;

private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
		
      edm::ESWatcher<SiPixelTemplateDBObjectRcd> SiPixelTemplateDBObjectWatcher_;

      std::string theTemplateCalibrationLocation;
      bool theDetailedTemplateDBErrorOutput;
      bool theFullTemplateDBOutput;

};

#endif
