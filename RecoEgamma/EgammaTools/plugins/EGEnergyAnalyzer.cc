// -*- C++ -*-
//
// Package:    EGEnergyAnalyzer
// Class:      EGEnergyAnalyzer
// 
/**\class EGEnergyAnalyzer EGEnergyAnalyzer.cc GBRWrap/EGEnergyAnalyzer/src/EGEnergyAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Josh Bendavid
//         Created:  Tue Nov  8 22:26:45 CET 2011
// $Id: EGEnergyAnalyzer.cc,v 1.3 2011/12/14 21:08:11 bendavid Exp $
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
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
//#include "CondCore/DBCommon/interface/CoralServiceManager.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergyCorrector.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


//
// class declaration
//

class EGEnergyAnalyzer : public edm::EDAnalyzer {
   public:
      explicit EGEnergyAnalyzer(const edm::ParameterSet&);
      ~EGEnergyAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      EGEnergyCorrector corfile;
      EGEnergyCorrector cordb;

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
EGEnergyAnalyzer::EGEnergyAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


EGEnergyAnalyzer::~EGEnergyAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
EGEnergyAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  if (!corfile.IsInitialized()) {
    corfile.Initialize(iSetup,"/afs/cern.ch/user/b/bendavid/cmspublic/gbrv3ph.root");
    //corfile.Initialize(iSetup,"wgbrph",true);
  }

  if (!cordb.IsInitialized()) {
    //cordb.Initialize(iSetup,"/afs/cern.ch/user/b/bendavid/cmspublic/regweights/gbrph.root");
    cordb.Initialize(iSetup,"wgbrph",true);
  }

  // get photon collection
  Handle<reco::PhotonCollection> hPhotonProduct;
  iEvent.getByLabel("photons",hPhotonProduct);
  
  EcalClusterLazyTools lazyTools(iEvent, iSetup, edm::InputTag("reducedEcalRecHitsEB"), 
                                 edm::InputTag("reducedEcalRecHitsEE"));  
  
  Handle<reco::VertexCollection> hVertexProduct;
  iEvent.getByLabel("offlinePrimaryVerticesWithBS", hVertexProduct);      
  
  for (reco::PhotonCollection::const_iterator it = hPhotonProduct->begin(); it!=hPhotonProduct->end(); ++it) {
    std::pair<double,double> corsfile = corfile.CorrectedEnergyWithError(*it, *hVertexProduct, lazyTools, iSetup);
    std::pair<double,double> corsdb = cordb.CorrectedEnergyWithError(*it, *hVertexProduct, lazyTools, iSetup);


    printf("file: default = %5f, correction = %5f, uncertainty = %5f\n", it->energy(),corsfile.first,corsfile.second);
    printf("db:   default = %5f, correction = %5f, uncertainty = %5f\n", it->energy(),corsdb.first,corsdb.second);

  }  




}


// ------------ method called once each job just before starting event loop  ------------
void 
EGEnergyAnalyzer::beginJob()
{



}

// ------------ method called once each job just after ending the event loop  ------------
void 
EGEnergyAnalyzer::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
EGEnergyAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
EGEnergyAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
EGEnergyAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
EGEnergyAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EGEnergyAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EGEnergyAnalyzer);
