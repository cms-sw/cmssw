// -*- C++ -*-
//
// Package:    WhatsItWatcherAnalyzer
// Class:      WhatsItWatcherAnalyzer
// 
/**\class WhatsItWatcherAnalyzer WhatsItWatcherAnalyzer.cc test/WhatsItWatcherAnalyzer/src/WhatsItWatcherAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 19:13:25 EDT 2005
// $Id: WhatsItWatcherAnalyzer.cc,v 1.2 2006/10/21 16:44:13 wmtan Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"


#include "FWCore/Integration/test/WhatsIt.h"
#include "FWCore/Integration/test/GadgetRcd.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/ESWatcher.h"

//
// class decleration
//

namespace edmtest {

class WhatsItWatcherAnalyzer : public edm::EDAnalyzer {
   public:
      explicit WhatsItWatcherAnalyzer(const edm::ParameterSet&);
      ~WhatsItWatcherAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
        void watch1(const GadgetRcd& );
        void watch2(const GadgetRcd& );
        
        edm::ESWatcher<GadgetRcd> watch1_;
        edm::ESWatcher<GadgetRcd> watch2_;
        edm::ESWatcher<GadgetRcd> watchBool_;

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
WhatsItWatcherAnalyzer::WhatsItWatcherAnalyzer(const edm::ParameterSet& /*iConfig*/):
  watch1_(this,&WhatsItWatcherAnalyzer::watch1),
  watch2_(boost::bind(&WhatsItWatcherAnalyzer::watch2,this,_1)),
  watchBool_()
{
   //now do what ever initialization is needed

}


WhatsItWatcherAnalyzer::~WhatsItWatcherAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
WhatsItWatcherAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup)
{
   bool w1 = watch1_.check(iSetup);
   bool w2 = watch2_.check(iSetup);
   bool w3 = watchBool_.check(iSetup);
   assert(w1 == w2);
   assert(w2 == w3 );
}

void
WhatsItWatcherAnalyzer::watch1(const GadgetRcd& iRcd)
{
  edm::ESHandle<edmtest::WhatsIt> pSetup;
  iRcd.get(pSetup);
  
  std::cout <<"watch1: WhatsIt "<<pSetup->a<<" changed"<<std::endl;
}

void
WhatsItWatcherAnalyzer::watch2(const GadgetRcd& iRcd)
{
  edm::ESHandle<WhatsIt> pSetup;
  iRcd.get(pSetup);
  
  std::cout <<"watch2: WhatsIt "<<pSetup->a<<" changed"<<std::endl;
}

}
using namespace edmtest;
//define this as a plug-in
DEFINE_FWK_MODULE(WhatsItWatcherAnalyzer);
