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
// $Id: OccupancyPlotter.cc,v 1.1 2011/08/05 09:19:38 slaunwhj Exp $
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
using std::vector;
using std::string;


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

  HLTConfigProvider hltConfig_;

  vector<std::string> triggersInSingleMuPD;
  
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
   using std::string;

   if (debugPrint) std::cout << "Inside analyze" << std::endl;

   if (testME) {
     testME->Fill(1);

   }


   // Access Trigger Results
   edm::Handle<edm::TriggerResults> triggerResults;
   iEvent.getByLabel(InputTag("TriggerResults","", "HLT"), triggerResults);
   if (!triggerResults.isValid()) return;

   if (debugPrint) std::cout << "Found triggerResults" << std::endl;


   edm::Handle<trigger::TriggerEvent>         aodTriggerEvent;   
   iEvent.getByLabel(InputTag("hltTriggerSummaryAOD", "", "HLT"), aodTriggerEvent);
   
   if ( !aodTriggerEvent.isValid() ) { 
     if (debugPrint) std::cout << "No AOD trigger summary found! Returning..."; 
     return; 
   }

   const TriggerObjectCollection objects = aodTriggerEvent->getObjects();

   for (unsigned int iPath = 0; iPath < triggersInSingleMuPD.size(); iPath++) {

     
     std::string pathName = triggersInSingleMuPD[iPath];

     std::cout << "Looking at path " << pathName << std::endl;
     
     unsigned int index = hltConfig_.triggerIndex(pathName);

     if (index < triggerResults->size()) {
       if(triggerResults->accept(index)){

         std::cout << "We fired path " << pathName << std::endl;

         // look up module labels for this path

         vector<std::string> modulesThisPath = hltConfig_.moduleLabels(pathName);

         if (debugPrint) std::cout << "Looping over module labels " << std::endl;

         // loop backward through module names
         for ( int iModule = (modulesThisPath.size()-1); iModule >= 0; iModule--) {

           if (debugPrint) std::cout << "Module name is " << modulesThisPath[iModule] << std::endl;

           // check to see if you have savetags information
           if (hltConfig_.saveTags(modulesThisPath[iModule])){

             if (debugPrint) std::cout << "This module " << modulesThisPath[iModule] <<" is a saveTags module of type " << hltConfig_.moduleType(modulesThisPath[iModule]) << std::endl;

             InputTag moduleWhoseResultsWeWant(modulesThisPath[iModule], "", "HLT");

             unsigned int indexOfModuleInAodTriggerEvent = aodTriggerEvent->filterIndex(moduleWhoseResultsWeWant);

             if ( indexOfModuleInAodTriggerEvent < aodTriggerEvent->sizeFilters() ) {
               const Keys &keys = aodTriggerEvent->filterKeys( indexOfModuleInAodTriggerEvent );
               if (debugPrint) std::cout << "Got Keys for index " << indexOfModuleInAodTriggerEvent <<", size of keys is " << keys.size() << std::endl;
               
               for ( size_t iKey = 0; iKey < keys.size(); iKey++ ){
                 TriggerObject foundObject = objects[keys[iKey]];
                 if (debugPrint) std::cout << "This object has (pt, eta, phi) = "
                                           << foundObject.pt()
                                           << ", " << foundObject.eta() << ", "
                                           << foundObject.phi() << std::endl;
               }// end for each key               
             }// end if filter in aodTriggerEvent


             // OK, we found the last module. No need to look at the others.
             // get out of the loop

             break;
           }// end if saveTags
         }//end for each module    
       }// end if accept
     }// end if index in triggerResults
   }// end for each path in PD


   

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
OccupancyPlotter::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{

  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, "HLT", changed)) {
    if(debugPrint)
      std::cout << "HLT config with process name " 
                << "HLT" << " successfully extracted" << std::endl;
  } else {
    if(debugPrint)
      std::cout << "Warning, didn't find process HLT" << std::endl;
  }

  vector<string> datasetNames =  hltConfig_.streamContent("A") ;
  for (unsigned int i=0;i<datasetNames.size();i++) {
    if (debugPrint) std::cout << "This is dataset " << datasetNames[i] <<std::endl;
    vector<string> datasetPaths = hltConfig_.datasetContent(datasetNames[i]);
    if (datasetNames[i] == "SingleMu"){      
      if (debugPrint) std::cout <<"Found SingleMu PD" << std::endl;
      triggersInSingleMuPD = datasetPaths;
    }    

  }// end of loop over dataset names

  
  
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
