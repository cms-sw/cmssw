#ifndef CondTools_SiPixel_SiPixelFakeTemplateDBSourceReader_h
#define CondTools_SiPixel_SiPixelFakeTemplateDBSourceReader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"

class SiPixelFakeTemplateDBSourceReader : public edm::EDAnalyzer {
   public:
      explicit SiPixelFakeTemplateDBSourceReader(const edm::ParameterSet&);
      ~SiPixelFakeTemplateDBSourceReader() override;


   private:
      void beginJob() override ;
			void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;
			
			edm::ESWatcher<SiPixelTemplateDBObjectRcd> SiPixelTemplateDBObjectWatcher_;

};

#endif
