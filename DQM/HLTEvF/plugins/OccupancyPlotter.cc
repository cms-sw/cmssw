// -*- C++ -*-
//
// Package:    OccupancyPlotter
// Class:      OccupancyPlotter
// 
/**\class OccupancyPlotter OccupancyPlotter.cc DQM/OccupancyPlotter/src/OccupancyPlotter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jason Michael Slaunwhite,512 1-008,+41227670494,
//         Created:  Fri Aug  5 10:34:47 CEST 2011
// $Id$
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

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"


//
// class declaration
//

using namespace edm;
using namespace trigger;



class OccupancyPlotter : public edm::EDAnalyzer {
   public:
      explicit OccupancyPlotter(const edm::ParameterSet&);
      ~OccupancyPlotter();

  //static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------


  bool debugPrint;

  std::string plotDirectoryName;

  DQMStore * dbe;

  MonitorElement* testME;
  
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
OccupancyPlotter::OccupancyPlotter(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

  debugPrint = true;

  if (debugPrint) std::cout << "Inside Constructor" << std::endl;

  plotDirectoryName = iConfig.getUntrackedParameter<std::string>("dirname", "HLT/Test");

  if (debugPrint) std::cout << "Got plot dirname = " << plotDirectoryName << std::endl;

}


OccupancyPlotter::~OccupancyPlotter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
OccupancyPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;


   if (debugPrint) std::cout << "Inside analyze" << std::endl;

   if (testME) {
     testME->Fill(1);

   }

}


// ------------ method called once each job just before starting event loop  ------------
void 
OccupancyPlotter::beginJob()
{

  if (debugPrint) std::cout << "Inside begin job" << std::endl; 

  dbe = Service<DQMStore>().operator->();

  if (dbe) {

    dbe->setCurrentFolder(plotDirectoryName);

  }

  testME = dbe->book1D("reportSummaryMap","report Summary Map",2,0,2);

  
}

// ------------ method called once each job just after ending the event loop  ------------
void 
OccupancyPlotter::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
OccupancyPlotter::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
OccupancyPlotter::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
OccupancyPlotter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
OccupancyPlotter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
// void
// OccupancyPlotter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   //The following says we do not know what parameters are allowed so do no validation
//   // Please change this to state exactly what you do use, even if it is no parameters
//   edm::ParameterSetDescription desc;
//   desc.setUnknown();
//   descriptions.addDefault(desc);
// }

//define this as a plug-in
DEFINE_FWK_MODULE(OccupancyPlotter);
