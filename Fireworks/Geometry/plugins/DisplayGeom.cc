// -*- C++ -*-
//
// Package:    DisplayGeom
// Class:      DisplayGeom
// 
/**\class DisplayGeom DisplayGeom.cc Reve/DisplayGeom/src/DisplayGeom.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris D Jones
//         Created:  Wed Sep 26 08:27:23 EDT 2007
// $Id: DisplayGeom.cc,v 1.21 2010/01/17 09:00:30 innocent Exp $
//
//

// system include files
#include <memory>
#include <iostream>
#include <sstream>

#include "TROOT.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TError.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "Fireworks/Geometry/interface/DisplayPluginFactory.h"


//
// class decleration
//
using namespace fireworks::geometry;
       
class DisplayGeom : public edm::EDAnalyzer {

   public:
      explicit DisplayGeom(const edm::ParameterSet&);
      ~DisplayGeom();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      int level_;
      bool verbose_;
      TApplication* app_;
      DisplayPlugin* plugin_;
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
DisplayGeom::DisplayGeom(const edm::ParameterSet& iConfig):
   level_(iConfig.getUntrackedParameter<int>("level",4)),
   verbose_(iConfig.getUntrackedParameter<bool>("verbose",false))
{
   //now do what ever initialization is needed

  std::cout <<" is batch "<<gROOT->IsBatch()<<std::endl;
  std::cout <<" display "<<gSystem->Getenv("DISPLAY")<<std::endl;

  const char* dummyArgvArray[] = {"cmsRun"};
  char** dummyArgv = const_cast<char**>(dummyArgvArray);
  int dummyArgc = 1;
  app_ = new TApplication("App", &dummyArgc, dummyArgv);
  assert(TApplication::GetApplications()->GetSize());
  
  gROOT->SetBatch(kFALSE);
  //TApplication* app = dynamic_cast<TApplication*>(TApplication::GetApplications()->First());
  //assert(app!=0);
  std::cout<<"calling NeedGraphicsLibs()"<<std::endl;
  TApplication::NeedGraphicsLibs();

  DisplayPluginFactory* factory = DisplayPluginFactory::get();
  plugin_ = factory->create("EveDisplayPlugin");
}


DisplayGeom::~DisplayGeom()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DisplayGeom::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::cout << "In the DisplayGeom::analyze method..." << std::endl;
   using namespace edm;

   //need to reset the Error handler to avoid error messages becoming exceptions
   ErrorHandlerFunc_t old = SetErrorHandler(DefaultErrorHandler);

   plugin_->run(iSetup);
   app_->Run(kTRUE);

   SetErrorHandler(old);

   // Exit from fireworks
   // gApplication
   //app->Terminate(0);

}


// ------------ method called once each job just before starting event loop  ------------
void 
DisplayGeom::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DisplayGeom::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(DisplayGeom);
