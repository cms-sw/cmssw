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
// $Id: PATUserDataTestModule.cc,v 1.5 2013/02/27 23:26:57 wmtan Exp $
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
#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include <string>

// BEGIN CRAZY WORKAROUND
namespace edm { using ::std::advance; }
// END CRAZY WORKAROUND
/*  Exmplanation of  the above crazy workaround:
    1) edm::Ptr uses 'advance' free function to locate a given item within a collection
    2) 'advance' is defined in  the std::namespace for std containers (e.g. vector)
       http://www.sgi.com/tech/stl/advance.html
    3) In edm::Ptr sources, we use 'advance'  without namespace prefix 
       http://cmslxr.fnal.gov/lxr/source/DataFormats/Common/interface/Ptr.h?v=CMSSW_2_1_10#214
    4) This normally work because the container is std:: so the free function is resolved in the
       correct namespace (don't ask me why or how it works; Ask Marc, Bill or read C++ standards)
    5) The hack 'namespace edm { using ::std::advance; }' imports std::advance into edm namespace,
       so that it works. Apparently the default implementation of std::advance is ok also for the
       iterator of the OwnVector
    6) Anyway, this should be solved upstream in OwnVector.h

            gpetruc
*/

//
// class decleration
//

class PATUserDataTestModule : public edm::EDProducer {
   public:
      explicit PATUserDataTestModule(const edm::ParameterSet&);
      ~PATUserDataTestModule();


   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------
      edm::InputTag muons_;
      std::string   label_;
      enum TestMode { TestRead, TestWrite, TestExternal };
      TestMode mode_;
};

//
// constructors and destructor
//
PATUserDataTestModule::PATUserDataTestModule(const edm::ParameterSet& iConfig):
  muons_(iConfig.getParameter<edm::InputTag>("muons")),
  label_(iConfig.existsAs<std::string>("label") ? iConfig.getParameter<std::string>("label") : ""),
  mode_( iConfig.getParameter<std::string>("mode") == "write" ? TestWrite : 
        (iConfig.getParameter<std::string>("mode") == "read"  ? TestRead  : 
         TestExternal
        ))

{
  if (mode_ != TestExternal) {
      produces<std::vector<pat::Muon> >();
  } else {
      produces<edm::ValueMap<int> >(label_);
      produces<edm::ValueMap<float> >();
      produces<edm::OwnVector<pat::UserData> >();
      produces<edm::ValueMap<edm::Ptr<pat::UserData> > >();
  }
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

   if (mode_ != TestExternal) {
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
               std::cout << "\tanswer:il= " << muon->userInt("answer:il") << std::endl;
               std::cout << "\tpi       = " << muon->userFloat("pi") << std::endl;
               std::cout << "\tmiss int = " << muon->userInt("missing int") << std::endl;
               std::cout << "\tmiss flt = " << muon->userFloat("missing flt") << std::endl;
               std::cout << "\t# u d    = " << muon->userDataNames().size() << std::endl;
               std::cout << "\tud nam[0]= " << muon->userDataNames()[0] << std::endl;
               std::cout << "\thas p4/2 = " << muon->hasUserData("half p4") << std::endl;
               if (muon->hasUserData("half p4")) {
                   std::cout << "\ttyp p4/2 = " << muon->userDataObjectType("half p4") << std::endl;
                   std::cout << "\tval p4/2 = " << (*muon->userData<reco::Particle::LorentzVector>("half p4")) << "  == " << (0.5*muon->p4()) << std::endl;
               }
               if (muon->hasUserData("tmp self")) {
                   std::cout << "\tself.pt  = " << muon->userData<pat::Muon>("tmp self")->pt() << "  == " << muon->pt() << std::endl;
               }
               if (muon->hasUserData("tmp crazy")) {
                   std::cout << "\tcrazy.siz= " << muon->userData<CrazyDataType>("tmp crazy")->second.size() << "  == 0 " << std::endl;
               }
               if (muon->hasUserData("halfP4")) {
                   std::cout << "\ttyp P4/2 = " << muon->userDataObjectType("halfP4") << std::endl;
                   std::cout << "\tval P4/2 = " << (*muon->userData<reco::Particle::LorentzVector>("halfP4")) << "  == " << (0.5*muon->p4()) << std::endl;
               }
           }
       }
       iEvent.put(output);
   } else { 
       using namespace std;
       Handle<View<reco::Muon> > recoMuons;
       iEvent.getByLabel(muons_, recoMuons);
       std::cout << "Got " << recoMuons->size() << " muons" << std::endl;

       vector<int> ints(recoMuons->size(), 42);
       auto_ptr<ValueMap<int> > answers(new ValueMap<int>());
       ValueMap<int>::Filler intfiller(*answers);
       intfiller.insert(recoMuons, ints.begin(), ints.end());
       intfiller.fill();
       iEvent.put(answers, label_);
       std::cout << "Filled in the answer" << std::endl;

       vector<float> floats(recoMuons->size(), 3.14);
       auto_ptr<ValueMap<float> > pis(new ValueMap<float>());
       ValueMap<float>::Filler floatfiller(*pis);
       floatfiller.insert(recoMuons, floats.begin(), floats.end());
       floatfiller.fill();
       iEvent.put(pis);
       std::cout << "Wrote useless floats into the event" << std::endl;

       auto_ptr<OwnVector<pat::UserData> > halfp4s(new OwnVector<pat::UserData>());
       for (size_t i = 0; i < recoMuons->size(); ++i) {
            halfp4s->push_back( pat::UserData::make( 0.5 * (*recoMuons)[i].p4() ) );
       }
       OrphanHandle<OwnVector<pat::UserData> > handle = iEvent.put(halfp4s);
       std::cout << "Wrote OwnVector of useless objects into the event" << std::endl;
       vector<Ptr<pat::UserData> > halfp4sPtr;
       for (size_t i = 0; i < recoMuons->size(); ++i) {
            // It is crucial to use the OrphanHandle here and not a RefProd from GetRefBeforePut
            halfp4sPtr.push_back(Ptr<pat::UserData>(handle, i));
       }
       std::cout << "   Made edm::Ptr<> to those useless objects" << std::endl;
       auto_ptr<ValueMap<Ptr<pat::UserData> > > vmhalfp4s(new ValueMap<Ptr<pat::UserData> >());
       ValueMap<Ptr<pat::UserData> >::Filler filler(*vmhalfp4s);
       filler.insert(recoMuons, halfp4sPtr.begin(), halfp4sPtr.end());
       filler.fill();
       std::cout << "   Filled the ValueMap of edm::Ptr<> to those useless objects" << std::endl;
       iEvent.put(vmhalfp4s);
       std::cout << "   Wrote the ValueMap of edm::Ptr<> to those useless objects" << std::endl;

       std::cout << "So long, and thanks for all the muons.\n" << std::endl;
   }

}

//define this as a plug-in
DEFINE_FWK_MODULE(PATUserDataTestModule);
