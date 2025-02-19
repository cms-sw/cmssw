// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      PATHIPhotonTestModule
// 
/**\class PATHIPhotonTestModule PATHIPhotonTestModule.cc PhysicsTools/PatAlgos/test/PATHIPhotonTestModule.cc

 Description: Test Photon isolation variables in PAT

 Implementation:
 
 this analyzer shows how to loop over PAT output. 
*/
//
// Original Author:  Yen-Jie Lee
//         Created:  Mon July 1 11:53:50 EST 2009
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
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/PatCandidates/interface/Photon.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <string>
#include "TNtuple.h"

namespace edm { using ::std::advance; }

//
// class decleration
//

class PATHIPhotonTestModule : public edm::EDAnalyzer {
   public:
      explicit PATHIPhotonTestModule(const edm::ParameterSet&);
      ~PATHIPhotonTestModule();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
   
      // ----------member data ---------------------------
      edm::InputTag photons_;
      std::string   label_;
      enum TestMode { TestRead, TestWrite, TestExternal };
      TestMode mode_;
      TNtuple *datatemp;
      edm::Service<TFileService> fs;
};

//
// constructors and destructor
//
PATHIPhotonTestModule::PATHIPhotonTestModule(const edm::ParameterSet& iConfig):
  photons_(iConfig.getParameter<edm::InputTag>("photons"))
{
}


PATHIPhotonTestModule::~PATHIPhotonTestModule()
{
}

// ------------ method called to for each event  ------------
void
PATHIPhotonTestModule::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::Handle<edm::View<pat::Photon> > photons;
   iEvent.getByLabel(photons_,photons);

   std::auto_ptr<std::vector<pat::Photon> > output(new std::vector<pat::Photon>());

   for (edm::View<pat::Photon>::const_iterator photon = photons->begin(), end = photons->end(); photon != end; ++photon) {
   	   
      Float_t var[100];
      
      int idx = 0;
      var[idx] = photon->et();
      idx++;
            
      for (int i=1;i<6;i++){
         var[idx]=photon->userFloat(Form("isoCC%d",i));
	 idx++;       
      }

      for (int i=1;i<6;i++){
         var[idx]=photon->userFloat(Form("isoCR%d",i));
	 idx++;    
      }

      for (int i=1;i<5;i++){
         for (int j=1;j<5;j++){
            var[idx]=photon->userFloat(Form("isoT%d%d",i,j));
	    idx++;     
	 }  
      }

      for (int i=1;i<5;i++){
         for (int j=1;j<5;j++){
            var[idx]=photon->userFloat(Form("isoDR%d%d",i,j));
	    idx++;     
	 }  
      }
 
      var[idx] = photon->e3x3();
      idx++;
      var[idx] = photon->e5x5();
      idx++;
      
      
      



datatemp->Fill(var);
   }

}

// ------------ method called once each job just before starting event loop  ------------
void 
PATHIPhotonTestModule::beginJob()
{

   datatemp = fs->make<TNtuple>("gammas", "photon candidate info", 
                                "et:"
				"cC1:cC2:cC3:cC4:cC5:cR1:cR2:cR3:cR4:cR5:"
				"T11:T12:T13:T14:"
				"T21:T22:T23:T24:"
				"T31:T32:T33:T34:"
				"T41:T42:T43:T44:"
				"dR11:dR12:dR13:dR14:"
				"dR21:dR22:dR23:dR24:"
				"dR31:dR32:dR33:dR34:"
				"dR41:dR42:dR43:dR44");
   }

// ------------ method called once each job just after ending the event loop  ------------
void 
PATHIPhotonTestModule::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PATHIPhotonTestModule);
