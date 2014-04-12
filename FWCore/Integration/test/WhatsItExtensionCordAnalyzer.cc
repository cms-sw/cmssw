// -*- C++ -*-
//
// Package:    WhatsItExtensionCordAnalyzer
// Class:      WhatsItExtensionCordAnalyzer
// 
/**\class WhatsItExtensionCordAnalyzer WhatsItExtensionCordAnalyzer.cc test/WhatsItExtensionCordAnalyzer/src/WhatsItExtensionCordAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 24 19:13:25 EDT 2005
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


//Here is the ExtensionCord/Outlet headers
#include "FWCore/Framework/interface/ESOutlet.h"
#include "FWCore/Utilities/interface/ExtensionCord.h"

//
// class decleration
//

namespace edmtest {

  class Last {
public:
    Last(const edm::ExtensionCord<WhatsIt>& iCord): cord_(iCord) {}
    void doIt() {
      std::cout <<"WhatsIt "<<cord_->a<<std::endl;
    }
private:
    edm::ExtensionCord<WhatsIt> cord_;
  };
  
  class Middle {
public:
    Middle(const edm::ExtensionCord<WhatsIt>& iCord): last_(iCord) {}
    void doIt() {
      last_.doIt();
    }
private:
    Last last_;
  };
  
class WhatsItExtensionCordAnalyzer : public edm::EDAnalyzer {
   public:
      explicit WhatsItExtensionCordAnalyzer(const edm::ParameterSet&);
      ~WhatsItExtensionCordAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
        edm::ExtensionCord<WhatsIt> cord_;
        Middle middle_;
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
WhatsItExtensionCordAnalyzer::WhatsItExtensionCordAnalyzer(const edm::ParameterSet& /*iConfig*/) :
cord_(),
middle_(cord_)
{
   //now do what ever initialization is needed

}


WhatsItExtensionCordAnalyzer::~WhatsItExtensionCordAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
WhatsItExtensionCordAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup)
{
  edm::ESOutlet<WhatsIt,GadgetRcd> outlet( iSetup, cord_ );
  
  middle_.doIt();
}

}
using namespace edmtest;
//define this as a plug-in
DEFINE_FWK_MODULE(WhatsItExtensionCordAnalyzer);
