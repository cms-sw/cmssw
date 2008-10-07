// -*- C++ -*-
//
// Package:    PatAlgos
// Class:      PATUserDataTestModule
// 
/**\class PATUserDataTestModule PATUserDataTestModule.cc PhysicsTools/PatAlgos/test/PATUserDataTestModule.cc

 Description: Test module for UserData in PAT

 Implementation:
 
 this analyzer shows how to loop over PAT output. 
*/
//
// Original Author:  Freya Blekman
//         Created:  Mon Apr 21 10:03:50 CEST 2008
// $Id: PATUserDataTestModule.cc,v 1.2 2008/05/13 10:25:05 fblekman Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/Common/interface/View.h"
#include <string>
//
// class decleration
//

class PATUserDataTestModule : public edm::EDProducer {
   public:
      explicit PATUserDataTestModule(const edm::ParameterSet&);
      ~PATUserDataTestModule();


   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
      edm::InputTag muons_;
      enum TestMode { TestRead, TestWrite };
      TestMode mode_;
};

//
// constructors and destructor
//
PATUserDataTestModule::PATUserDataTestModule(const edm::ParameterSet& iConfig):
  muons_(iConfig.getParameter<edm::InputTag>("muons")),
  mode_(iConfig.getParameter<std::string>("mode") == "write" ? TestWrite : TestRead)
{
  produces<std::vector<pat::Muon> >();
}


PATUserDataTestModule::~PATUserDataTestModule()
{
}

// ------------ method called to for each event  ------------
void
PATUserDataTestModule::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // I need to define a type for which we don't have a dictionary in CMSSW. I hope this is crazy enough.
   typedef std::pair<std::map<std::string,pat::Muon>, std::vector<math::XYZVector> > CrazyDataType;

   edm::Handle<edm::View<pat::Muon> > muons;
   iEvent.getByLabel(muons_,muons);

   std::auto_ptr<std::vector<pat::Muon> > output(new std::vector<pat::Muon>());

   for (edm::View<pat::Muon>::const_iterator muon = muons->begin(), end = muons->end(); muon != end; ++muon) {
       if (mode_ == TestWrite) {
            pat::Muon myMuon = *muon; // copy
            myMuon.addUserInt("answer", 42);
            myMuon.addUserFloat("pi", 3.14);
            myMuon.addUserData("half p4", 0.5*muon->p4());

            //// This shoud throw an exception because we don't have the wrapper type
            //myMuon.addUserData("self", *muon);
            //// This should throw an exception because we don't have the dictionary (I hope)
            // myMuon.addUserData("crazy", CrazyDataType());
            //// These instead should not throw an exception as long as they don't get saved on disk
            // myMuon.addUserData("tmp self", *muon, true);
            // myMuon.addUserData("tmp crazy", CrazyDataType(), true);

            output->push_back(myMuon);
       } else {
            std::cout << "Muon #" << (muon - muons->begin()) << ":" << std::endl;
            std::cout << "\tanswer   = " << muon->userInt("answer") << std::endl;
            std::cout << "\tpi       = " << muon->userFloat("pi") << std::endl;
            std::cout << "\tmiss int = " << muon->userInt("missing int") << std::endl;
            std::cout << "\tmiss flt = " << muon->userFloat("missing flt") << std::endl;
            std::cout << "\t# u d    = " << muon->userDataNames().size() << std::endl;
            std::cout << "\tud nam[0]= " << muon->userDataNames()[0] << std::endl;
            std::cout << "\thas p4/2 = " << muon->hasUserData("half p4") << std::endl;
            std::cout << "\ttyp p4/2 = " << muon->userDataObjectType("half p4") << std::endl;
            std::cout << "\ttyp p4/2 = " << (*muon->userData<reco::Particle::LorentzVector>("half p4")) << "  == " << (0.5*muon->p4()) << std::endl;
            if (muon->hasUserData("tmp self")) {
                std::cout << "\tself.pt  = " << muon->userData<pat::Muon>("tmp self")->pt() << "  == " << muon->pt() << std::endl;
            }
            if (muon->hasUserData("tmp crazy")) {
                std::cout << "\tcrazy.siz= " << muon->userData<CrazyDataType>("tmp crazy")->second.size() << "  == 0 " << std::endl;
            }
       }
   }

   iEvent.put(output);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PATUserDataTestModule);
