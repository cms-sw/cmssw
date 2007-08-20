#ifndef CalibTracker_SiPixelErrorEstimation_PxCPEdbUploader_h
#define CalibTracker_SiPixelErrorEstimation_PxCPEdbUploader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PxCPEdbUploader : public edm::EDAnalyzer {
   public:
      explicit PxCPEdbUploader(const edm::ParameterSet&);
      ~PxCPEdbUploader();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
     	std::string theFileName ;

};

#endif
