#ifndef CalibTracker_SiPixelErrorEstimation_PxCPEdbReader_h
#define CalibTracker_SiPixelErrorEstimation_PxCPEdbReader_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


class PxCPEdbReader : public edm::EDAnalyzer {
   public:
      explicit PxCPEdbReader(const edm::ParameterSet&);
      ~PxCPEdbReader();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

};

#endif
