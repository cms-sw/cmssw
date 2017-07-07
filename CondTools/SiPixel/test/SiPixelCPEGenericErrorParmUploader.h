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
      ~SiPixelCPEGenericErrorParmUploader() override;


   private:
      void beginJob() override ;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;
			edm::FileInPath theFileName ;
			double theVersion;

};

#endif
