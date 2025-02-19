#ifndef CondTools_SiPixel_SiPixelCPEGenericErrorParmUploader_h
#define CondTools_SiPixel_SiPixelCPEGenericErrorParmUploader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiPixelCPEGenericErrorParmUploader : public edm::EDAnalyzer {
   public:
      explicit SiPixelCPEGenericErrorParmUploader(const edm::ParameterSet&);
      ~SiPixelCPEGenericErrorParmUploader();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
			edm::FileInPath theFileName ;
			double theVersion;

};

#endif
