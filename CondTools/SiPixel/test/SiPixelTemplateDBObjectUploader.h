#ifndef CondTools_SiPixel_SiPixelTemplateDBObjectUploader_h
#define CondTools_SiPixel_SiPixelTemplateDBObjectUploader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiPixelTemplateDBObjectUploader : public edm::EDAnalyzer {
   public:
      explicit SiPixelTemplateDBObjectUploader(const edm::ParameterSet&);
      ~SiPixelTemplateDBObjectUploader();

			typedef std::vector<std::string> vstring;

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
     	vstring theFileNums ;
			float theVersion;

};

#endif
