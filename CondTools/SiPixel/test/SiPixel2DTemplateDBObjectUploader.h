#ifndef CondTools_SiPixel_SiPixel2DTemplateDBObjectUploader_h
#define CondTools_SiPixel_SiPixel2DTemplateDBObjectUploader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"

class SiPixel2DTemplateDBObjectUploader : public edm::EDAnalyzer {
   public:
      explicit SiPixel2DTemplateDBObjectUploader(const edm::ParameterSet&);
      ~SiPixel2DTemplateDBObjectUploader();

			typedef std::vector<std::string> vstring;

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
			
			vstring theTemplateCalibrations ;
			std::string theTemplateBaseString;
			float theVersion;
			float theMagField;
			std::vector<uint32_t> theDetIds;
			std::vector<uint32_t> theTemplIds;
		
};

#endif
