// -*- C++ -*-
//
// Package:    EmDQMFeeder
// Class:      EmDQMFeeder
// 
/**\class EmDQMFeeder EmDQMFeeder.cc HLTriggerOffline/Egamma/src/EmDQMFeeder.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Reis,40 4-B24,+41227671567,
//         Created:  Tue Mar 15 12:24:11 CET 2011
// $Id$
//
//


// system include files
#include <memory>
#include <iostream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
//
// class declaration
//

class EmDQMFeeder : public edm::EDAnalyzer {
   public:
      explicit EmDQMFeeder(const edm::ParameterSet&);
      ~EmDQMFeeder();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      void beginRun(edm::Run const & iRun, edm::EventSetup const& iSetup);

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      //virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------

      std::string processName_; // process name of (HLT) process for which to get HLT configuration
      /// The instance of the HLTConfigProvider as a data member
      HLTConfigProvider hltConfig_;

      std::vector<std::vector<std::string> > findEgammaPaths();
      std::vector<std::string> getFilterModules(const std::string&);
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
EmDQMFeeder::EmDQMFeeder(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

   processName_ = iConfig.getParameter<std::string>("processname");
}


EmDQMFeeder::~EmDQMFeeder()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
EmDQMFeeder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;



#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
EmDQMFeeder::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EmDQMFeeder::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
EmDQMFeeder::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
   bool changed(true);
   if (hltConfig_.init(iRun, iSetup, processName_, changed)) {
      // if init returns TRUE, initialisation has succeeded!

      // Output general information on the menu
      std::cout << "inited=" << hltConfig_.inited() << std::endl;
      std::cout << "changed=" << hltConfig_.changed() << std::endl;
      std::cout << "processName=" << hltConfig_.processName() << std::endl;
      std::cout << "tableName=" << hltConfig_.tableName() << std::endl;
      std::cout << "size=" << hltConfig_.size() << std::endl << std::endl;

      // All electron and photon paths
      // index 0: electron
      //       1: double electron
      //       2: photon
      //       3: double photon
      std::vector<std::vector<std::string> > egammaPaths = findEgammaPaths();
      //std::cout << "Found " << egammaPaths[0].size() << " single electron paths" << std::endl;
      //std::cout << "Found " << egammaPaths[1].size() << " double electron paths" << std::endl;
      //std::cout << "Found " << egammaPaths[2].size() << " single photon paths" << std::endl;
      //std::cout << "Found " << egammaPaths[3].size() << " double photon paths" << std::endl;

      //print filters of all egamma paths
      std::vector<std::string> filterModules;
      for (unsigned int j=0; j < egammaPaths.size() ; j++) {
         for (unsigned int i=0; i < egammaPaths.at(j).size() ; i++) {
            std::cout << i<< std::endl;
            filterModules = getFilterModules(egammaPaths.at(j).at(i));
         }
      }

      edm::ParameterSet pathPSet;

      pathPSet = hltConfig_.modulePSet(filterModules.at(5));
      std::cout << pathPSet.dump() << std::endl;

      if (changed) {
         // The HLT config has actually changed wrt the previous Run, hence rebook your
         // histograms or do anything else dependent on the revised HLT config
      }
   } else {
      // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
      // with the file and/or code and needs to be investigated!
      edm::LogError("EmDQMFeeder") << " HLT config extraction failure with process name " << processName_;
      // In this case, all access methods will return empty values!
   }
}

// ------------ method called when ending the processing of a run  ------------
void 
EmDQMFeeder::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
EmDQMFeeder::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
EmDQMFeeder::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EmDQMFeeder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
   //The following says we do not know what parameters are allowed so do no validation
   // Please change this to state exactly what you do use, even if it is no parameters

   // see: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideConfigurationValidationAndHelp
   edm::ParameterSetDescription desc;
   desc.setUnknown();
   descriptions.addDefault(desc);
}

std::vector<std::vector<std::string> >
EmDQMFeeder::findEgammaPaths()
{
   std::vector<std::vector<std::string> > Paths(4);
   // Loop over all paths in the menu
   for (unsigned int i=0; i<hltConfig_.size(); i++) {

      std::string path = hltConfig_.triggerName(i);

      // Find electron and photon paths
      if (int(path.find("HLT_")) == 0) {    // Path should start with 'HLT_'
         if (path.find("HLT_Ele") != std::string::npos) {
            Paths[0].push_back(path);
            //std::cout << "Electron ";
         }
         else if (path.find("HLT_DoubleEle") != std::string::npos) {
            Paths[1].push_back(path);
            //std::cout << "DoubleElectron ";
         }
         else if (path.find("HLT_Photon") != std::string::npos) {
            Paths[2].push_back(path);
            //std::cout << "Photon ";
         }
         else if (path.find("HLT_DoublePhoton") != std::string::npos) {
            Paths[3].push_back(path);
            //std::cout << "DoublePhoton ";
         }
      }
      //std::cout << i << " triggerName: " << path << " containing " << hltConfig_.size(i) << " modules."<< std::endl;
   }
   std::cout << std::endl;
   return Paths;
}

std::vector<std::string>
EmDQMFeeder::getFilterModules(const std::string& path)
{
   std::vector<std::string> filters;

   std::cout << "Pathname: " << path << std::endl;

   // Loop over all modules in the path
   for (unsigned int i=0; i<hltConfig_.size(path); i++) {

      std::string module = hltConfig_.moduleLabel(path, i);
      std::string moduleType = hltConfig_.moduleType(module);
      std::string moduleEDMType = hltConfig_.moduleEDMType(module);

      // Find filters
      if (moduleEDMType.find("EDFilter") != std::string::npos || moduleType.find("Filter") != std::string::npos) {  // older samples may not have EDMType data included
         filters.push_back(module);
         std::cout << i << "    moduleLabel: " << module << "    moduleType: " << moduleType << "    moduleEDMType: " << moduleEDMType << std::endl;
      }
   }
   return filters;
}

//define this as a plug-in
DEFINE_FWK_MODULE(EmDQMFeeder);
