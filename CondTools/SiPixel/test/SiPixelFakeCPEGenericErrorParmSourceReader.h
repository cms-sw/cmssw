#ifndef CondTools_SiPixel_SiPixelFakeCPEGenericErrorParmSourceReader_h
#define CondTools_SiPixel_SiPixelFakeCPEGenericErrorParmSourceReader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/SiPixelCPEGenericErrorParmRcd.h"

class SiPixelFakeCPEGenericErrorParmSourceReader : public edm::EDAnalyzer {
   public:
      explicit SiPixelFakeCPEGenericErrorParmSourceReader(const edm::ParameterSet&);
      ~SiPixelFakeCPEGenericErrorParmSourceReader();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

			edm::ESWatcher<SiPixelCPEGenericErrorParmRcd> SiPixelCPEGenericErrorParmWatcher_;
};

#endif
