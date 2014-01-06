
// -*- C++ -*-
//
// Package:    GBRWrapperMaker
// Class:      GBRWrapperMaker
// 
/**\class GBRWrapperMaker GBRWrapperMaker.cc GBRWrap/GBRWrapperMaker/src/GBRWrapperMaker.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Josh Bendavid
//         Created:  Tue Nov  8 22:26:45 CET 2011
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TFile.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
//#include "CondCore/DBCommon/interface/CoralServiceManager.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"


//
// class declaration
//

class GBRWrapperMaker : public edm::EDAnalyzer {
   public:
      explicit GBRWrapperMaker(const edm::ParameterSet&);
      ~GBRWrapperMaker();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
GBRWrapperMaker::GBRWrapperMaker(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


GBRWrapperMaker::~GBRWrapperMaker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
GBRWrapperMaker::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;



// #ifdef THIS_IS_AN_EVENT_EXAMPLE
//    Handle<ExampleData> pIn;
//    iEvent.getByLabel("example",pIn);
// #endif
//    
// #ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
//    ESHandle<SetupData> pSetup;
//    iSetup.get<SetupRecord>().get(pSetup);
// #endif
//from Josh:
  TFile *infile = new TFile("eleEnergyRegWeights_WithSubClusters_VApr15.root","READ");

  printf("load forest\n");


  GBRForest *p4comb = (GBRForest*)infile->Get("CombinationWeight");
  //GBRForest *gbbrebvar = (GBRForest*)infile->Get("EBUncertainty");
  //GBRForest *gbree = (GBRForest*)infile->Get("EECorrection");
  //GBRForest *gbreevar = (GBRForest*)infile->Get("EEUncertainty");




//from Rishi
  /*
  TFile *infile_PFLC = new TFile("/afs/cern.ch/user/r/rpatel/ConvXml/TMVARegression_BDTG_PFClusterCorr.root","READ");
  TFile *infile_PFGC = new TFile("/afs/cern.ch/user/r/rpatel/ConvXml/TMVARegression_BDTG_PFGlobalCorr.root","READ");
  TFile *infile_PFRes = new TFile("/afs/cern.ch/user/r/rpatel/ConvXml/TMVARegression_BDTG_PFRes.root ","READ");
  
  GBRForest *gbrLC = (GBRForest*)infile_PFLC->Get("GBRForest");
  GBRForest *gbrGC = (GBRForest*)infile_PFGC->Get("GBRForest");
  GBRForest *gbrRes = (GBRForest*)infile_PFRes->Get("GBRForest");
  */
  
  
  
  
  printf("made objects\n");
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
       
 poolDbService->writeOne( p4comb, poolDbService->beginOfTime(),
                                                "gedelectron_p4combination"  );
 //poolDbService->writeOne( gbrebvar, poolDbService->beginOfTime(),
 //"wgbrph_EBUncertainty"  );
 //poolDbService->writeOne( gbree, poolDbService->beginOfTime(),
 //"pfecalsc_EECorrection"  );
 //poolDbService->writeOne( gbreevar, poolDbService->beginOfTime(),
 //		  "wgbrph_EEUncertainty"  );
    
 //poolDbService->writeOne( gbrLC, poolDbService->beginOfTime(),
 //		     "wgbrph_PFLCCorrection"  );
 // poolDbService->writeOne( gbrGC, poolDbService->beginOfTime(),
 //		     "wgbrph_PFGlobalCorrection"  );
 // poolDbService->writeOne( gbrRes, poolDbService->beginOfTime(),
 //		     "wgbrph_PFResolution"  );
  
  }


}


// ------------ method called once each job just before starting event loop  ------------
void 
GBRWrapperMaker::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GBRWrapperMaker::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
GBRWrapperMaker::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
GBRWrapperMaker::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
GBRWrapperMaker::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
GBRWrapperMaker::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
GBRWrapperMaker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GBRWrapperMaker);
