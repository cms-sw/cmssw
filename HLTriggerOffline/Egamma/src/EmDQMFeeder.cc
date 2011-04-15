// -*- C++ -*-
//
// Package:    EmDQMFeeder
// Class:      EmDQMFeeder
// 
/**\class EmDQMFeeder EmDQMFeeder.cc HLTriggerOffline/Egamma/src/EmDQMFeeder.cc

 Description: Reads the trigger menu and calls EmDQM with generated parameter sets for each Egamma path

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Reis,40 4-B24,+41227671567,
//         Created:  Tue Mar 15 12:24:11 CET 2011
// $Id: EmDQMFeeder.cc,v 1.5 2011/04/15 11:50:37 treis Exp $
//
//


// system include files
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTriggerOffline/Egamma/interface/EmDQM.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
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

      const edm::ParameterSet& iConfig;

      std::string processName_; // process name of (HLT) process for which to get HLT configuration
      /// The instance of the HLTConfigProvider as a data member
      HLTConfigProvider hltConfig_;

      std::vector<std::vector<std::string> > findEgammaPaths();
      std::vector<std::string> getFilterModules(const std::string&);
      double getPrimaryEtCut(const std::string&);
     
      edm::ParameterSet makePSetForL1SeedFilter(const std::string&);
      edm::ParameterSet makePSetForL1SeedToSuperClusterMatchFilter(const std::string&);
      edm::ParameterSet makePSetForEtFilter(const std::string&);
      edm::ParameterSet makePSetForOneOEMinusOneOPFilter(const std::string&);
      edm::ParameterSet makePSetForPixelMatchFilter(const std::string&);
      edm::ParameterSet makePSetForEgammaGenericFilter(const std::string&, const std::string&);
      edm::ParameterSet makePSetForEgammaGenericQuadraticFilter(const std::string&, const std::string&);
      edm::ParameterSet makePSetForElectronGenericFilter(const std::string&, const std::string&);

      std::vector<EmDQM*> emDQMmodules;

      static const unsigned TYPE_SINGLE_ELE = 0;
      static const unsigned TYPE_DOUBLE_ELE = 1;
      static const unsigned TYPE_SINGLE_PHOTON = 2;
      static const unsigned TYPE_DOUBLE_PHOTON = 3;


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
EmDQMFeeder::EmDQMFeeder(const edm::ParameterSet& iConfig_) :
  iConfig(iConfig_)
{
   //now do what ever initialization is needed
   processName_ = iConfig_.getParameter<std::string>("processname");
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
 
   for (unsigned i = 0; i < emDQMmodules.size(); ++i) {
      emDQMmodules[i]->analyze(iEvent, iSetup);
   }

// #ifdef THIS_IS_AN_EVENT_EXAMPLE
//    Handle<ExampleData> pIn;
//    iEvent.getByLabel("example",pIn);
// #endif
//    
// #ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
//    ESHandle<SetupData> pSetup;
//    iSetup.get<SetupRecord>().get(pSetup);
// #endif
}


// ------------ method called once each job just before starting event loop  ------------
void 
EmDQMFeeder::beginJob()
{
//  std::cout << "EmDQMFeeder: beginJob" << std::endl;
//  for (unsigned i = 0; i < emDQMmodules.size(); ++i) {
//    std::cout << "EmDQM: beginJob for filter " << i  << std::endl;
//    emDQMmodules[i]->beginJob();
//  }
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EmDQMFeeder::endJob() 
{
  //std::cout << "EmDQMFeeder: endJob" << std::endl;
  for (unsigned i = 0; i < emDQMmodules.size(); ++i) {
    //std::cout << "EmDQM: endJob for filter " << i << std::endl;
    emDQMmodules[i]->endJob();
  }
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
      std::vector<std::vector<std::string> > egammaPaths = findEgammaPaths();
      //std::cout << "Found " << egammaPaths[TYPE_SINGLE_ELE].size() << " single electron paths" << std::endl;
      //std::cout << "Found " << egammaPaths[TYPE_DOUBLE_ELE].size() << " double electron paths" << std::endl;
      //std::cout << "Found " << egammaPaths[TYPE_SINGLE_PHOTON].size() << " single photon paths" << std::endl;
      //std::cout << "Found " << egammaPaths[TYPE_DOUBLE_PHOTON].size() << " double photon paths" << std::endl;

      std::vector<std::string> filterModules;

      for (unsigned int j=0; j < egammaPaths.size() ; j++) {

         for (unsigned int i=0; i < egammaPaths.at(j).size() ; i++) {
            // get pathname of this trigger 
	    const std::string pathName = egammaPaths.at(j).at(i);
            std::cout << "Path: " << pathName << std::endl;

            // get filters of the current path
            filterModules = getFilterModules(pathName);

            //--------------------	   
	    edm::ParameterSet paramSet;

	    paramSet.addParameter("@module_label", pathName + "_DQM");
	    paramSet.addParameter("triggerobject", iConfig.getParameter<edm::InputTag>("triggerobject"));

 	    //paramSet.addParameter("reqNum", iConfig.getParameter<unsigned int>("reqNum"));
	    //paramSet.addParameter("pdgGen", iConfig.getParameter<int>("pdgGen"));
	    paramSet.addParameter("genEtaAcc", iConfig.getParameter<double>("genEtaAcc"));
	    paramSet.addParameter("genEtAcc", iConfig.getParameter<double>("genEtAcc"));

	    // plotting parameters (untracked because they don't affect the physics)
            try {
               paramSet.addUntrackedParameter("genEtMin", getPrimaryEtCut(pathName));
            }
            catch (...) {
               std::cout << "Exception caught while generating the parameter set of the path '" << pathName << "' for use in EmDQM.  Will not include this path in the validation." << std::endl;
               continue;
            }
	    paramSet.addUntrackedParameter("PtMin", iConfig.getUntrackedParameter<double>("PtMin",0.));
	    paramSet.addUntrackedParameter("PtMax", iConfig.getUntrackedParameter<double>("PtMax",1000.));
	    paramSet.addUntrackedParameter("EtaMax", iConfig.getUntrackedParameter<double>("EtaMax", 2.7));
	    paramSet.addUntrackedParameter("PhiMax", iConfig.getUntrackedParameter<double>("PhiMax", 3.15));
	    paramSet.addUntrackedParameter("Nbins", iConfig.getUntrackedParameter<unsigned int>("Nbins",40));
	    paramSet.addUntrackedParameter("minEtForEtaEffPlot", iConfig.getUntrackedParameter<unsigned int>("minEtForEtaEffPlot", 15));
	    paramSet.addUntrackedParameter("useHumanReadableHistTitles", iConfig.getUntrackedParameter<bool>("useHumanReadableHistTitles", false));

	    //preselction cuts 
            switch (j) {
               case TYPE_SINGLE_ELE:
 	          paramSet.addParameter<unsigned>("reqNum", 1);
	          paramSet.addParameter<int>("pdgGen", 11);
	          paramSet.addParameter<edm::InputTag>("cutcollection", edm::InputTag("fiducialWenu"));
                  paramSet.addParameter<int>("cutnum", 1);
                  break;
               case TYPE_DOUBLE_ELE:           
 	          paramSet.addParameter<unsigned>("reqNum", 2);
	          paramSet.addParameter<int>("pdgGen", 11);
                  paramSet.addParameter<edm::InputTag>("cutcollection", edm::InputTag("fiducialZee"));
	          paramSet.addParameter<int>("cutnum", 2);
                  break;
               case TYPE_SINGLE_PHOTON:           
 	          paramSet.addParameter<unsigned>("reqNum", 1);
	          paramSet.addParameter<int>("pdgGen", 22);
                  paramSet.addParameter<edm::InputTag>("cutcollection", edm::InputTag("fiducialGammaJet"));
	          paramSet.addParameter<int>("cutnum", 1);
                  break;
               case TYPE_DOUBLE_PHOTON:           
 	          paramSet.addParameter<unsigned>("reqNum", 2);
	          paramSet.addParameter<int>("pdgGen", 22);
                  paramSet.addParameter<edm::InputTag>("cutcollection", edm::InputTag("fiducialDiGamma"));
	          paramSet.addParameter<int>("cutnum", 2);
            }
	    //--------------------

            try {
               std::vector<edm::ParameterSet> filterVPSet;

	       // loop over filtermodules of current trigger path
               for (std::vector<std::string>::iterator filter = filterModules.begin(); filter != filterModules.end(); ++filter) {
	          std::string moduleType = hltConfig_.modulePSet(*filter).getParameter<std::string>("@module_type");
	          std::string moduleLabel = hltConfig_.modulePSet(*filter).getParameter<std::string>("@module_label");

                  // first check if it is one filter we are not interrested in
                  if (moduleType == "Pythia6GeneratorFilter" ||
                      moduleType == "HLTTriggerTypeFilter" ||
                      moduleType == "HLTLevel1Activity" ||
                      moduleType == "HLTPrescaler" ||
                      moduleType == "HLTBool")
                     continue;

                  // now check for the known filter types
                  if (moduleType == "HLTLevel1GTSeed") {
                     filterVPSet.push_back(makePSetForL1SeedFilter(moduleLabel));
                     continue;
                  }
                  if (moduleType == "HLTEgammaL1MatchFilterRegional") {
                     filterVPSet.push_back(makePSetForL1SeedToSuperClusterMatchFilter(moduleLabel));
                     continue;
                  }
                  if (moduleType == "HLTEgammaEtFilter") {
                     filterVPSet.push_back(makePSetForEtFilter(moduleLabel));
                     continue;
                  }
                  if (moduleType == "HLTElectronOneOEMinusOneOPFilterRegional") {
                     filterVPSet.push_back(makePSetForOneOEMinusOneOPFilter(moduleLabel));
                     continue;
                  }
                  if (moduleType == "HLTElectronPixelMatchFilter") {
                     filterVPSet.push_back(makePSetForPixelMatchFilter(moduleLabel));
                     continue;
                  }
                  if (moduleType == "HLTEgammaGenericFilter") {
                     filterVPSet.push_back(makePSetForEgammaGenericFilter(pathName, moduleLabel));
                     continue;
                  }
                  //if (moduleType == "HLTEgammaGenericQuadraticFilter") {
                  //   filterVPSet.push_back(makePSetForEgammaGenericQuadraticFilter(pathName, moduleLabel));
                  //   continue;
                  //}
                  if (moduleType == "HLTElectronGenericFilter") {
                     filterVPSet.push_back(makePSetForElectronGenericFilter(pathName, moduleLabel));
                     continue;
                  }
                  std::cout << "No parameter set for filter '" << moduleLabel << "' with filter type '" << moduleType << "' added. Module will not be analyzed." << std::endl;
               } // end loop over filter modules of current trigger path

               paramSet.addParameter<std::vector<edm::ParameterSet> >("filters", filterVPSet);
            }
            catch (...) {
               std::cout << "Exception caught while generating the parameter set of the path '" << pathName << "' for use in EmDQM.  Will not include this path in the validation." << std::endl;
               continue;
            }

            // dump generated parameter set
            //std::cout << paramSet.dump() << std::endl;

	    emDQMmodules.push_back(new EmDQM(paramSet));

	    // emDQMmodules.back()->beginRun(iRun, iSetup);
	    emDQMmodules.back()->beginJob();
         } // loop over all paths of this analysis type

      } // loop over analysis types (single ele etc.)

      if (changed) {
         // The HLT config has actually changed wrt the previous Run, hence rebook your
         // histograms or do anything else dependent on the revised HLT config
      }
   } else {
      // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
      // with the file and/or code and needs to be investigated!
      edm::LogError("EmDQMFeeder") << " HLT config extraction failure with process name '" << processName_ << "'.";
      // In this case, all access methods will return empty values!
   }
}

// ------------ method called when ending the processing of a run  ------------
void 
EmDQMFeeder::endRun(edm::Run const&iEvent, edm::EventSetup const&iSetup)
{
//  for (unsigned i = 0; i < emDQMmodules.size(); ++i)
//     emDQMmodules[i]->endRun(iEvent, iSetup);
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
EmDQMFeeder::beginLuminosityBlock(edm::LuminosityBlock const&lumi, edm::EventSetup const&iSetup)
{
//  for (unsigned i = 0; i < emDQMmodules.size(); ++i)
//     emDQMmodules[i]->beginLuminosityBlock(lumi, iSetup);
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
EmDQMFeeder::endLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& iSetup)
{
//  for (unsigned i = 0; i < emDQMmodules.size(); ++i)
//     emDQMmodules[i]->endLuminosityBlock(lumi, iSetup);
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

//----------------------------------------------------------------------

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
            Paths[TYPE_SINGLE_ELE].push_back(path);
            //std::cout << "Electron ";
         }
         else if (path.find("HLT_DoubleEle") != std::string::npos) {
            Paths[TYPE_DOUBLE_ELE].push_back(path);
            //std::cout << "DoubleElectron ";
         }
         else if (path.find("HLT_Photon") != std::string::npos) {
            Paths[TYPE_SINGLE_PHOTON].push_back(path);
            //std::cout << "Photon ";
         }
         else if (path.find("HLT_DoublePhoton") != std::string::npos) {
            Paths[TYPE_DOUBLE_PHOTON].push_back(path);
            //std::cout << "DoublePhoton ";
         }
      }
      //std::cout << i << " triggerName: " << path << " containing " << hltConfig_.size(i) << " modules."<< std::endl;
   }
   return Paths;
}

//----------------------------------------------------------------------

std::vector<std::string>
EmDQMFeeder::getFilterModules(const std::string& path)
{
   std::vector<std::string> filters;

   //std::cout << "Pathname: " << path << std::endl;

   // Loop over all modules in the path
   for (unsigned int i=0; i<hltConfig_.size(path); i++) {

      std::string module = hltConfig_.moduleLabel(path, i);
      std::string moduleType = hltConfig_.moduleType(module);
      std::string moduleEDMType = hltConfig_.moduleEDMType(module);

      // Find filters
      if (moduleEDMType == "EDFilter" || moduleType.find("Filter") != std::string::npos) {  // older samples may not have EDMType data included
         filters.push_back(module);
         //std::cout << i << "    moduleLabel: " << module << "    moduleType: " << moduleType << "    moduleEDMType: " << moduleEDMType << std::endl;
      }
   }
   return filters;
}

//----------------------------------------------------------------------

double
EmDQMFeeder::getPrimaryEtCut(const std::string& path)
{
   double minEt = -1;

   boost::regex reg("^HLT_.*?([[:digit:]]+).*");

   boost::smatch what;
   if (boost::regex_match(path, what, reg, boost::match_extra))
   {
     minEt = boost::lexical_cast<double>(what[1]); 
   }
   else {
     edm::LogError("EmDQMFeeder") << "Unable to determine a minimum Et from the path name '" << path << "'.";
     throw "noMinEt";
   }

   return minEt;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForL1SeedFilter(const std::string& moduleName)
{
  // generates a PSet to analyze the behaviour of an L1 seed.
  //
  // moduleName is the name of the HLT module which filters
  // on the L1 seed.
  edm::ParameterSet retPSet;
  
  retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerL1NoIsoEG);

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForL1SeedToSuperClusterMatchFilter(const std::string& moduleName)
{
  // generates a PSet to analyze the behaviour of L1 to supercluster match filter.
  //
  // moduleName is the name of the HLT module which requires the match
  // between supercluster and L1 seed.
  //
  edm::ParameterSet retPSet;
  
  retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForEtFilter(const std::string& moduleName)
{
  // generates a PSet for the Egamma DQM analyzer for the Et filter
  edm::ParameterSet retPSet;
  
  retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForOneOEMinusOneOPFilter(const std::string& moduleName)
{
  edm::ParameterSet retPSet;
  
  retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerElectron);

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForPixelMatchFilter(const std::string& moduleName)
{
  edm::ParameterSet retPSet;
  
  retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForEgammaGenericFilter(const std::string& pathName, const std::string& moduleName)
{
  edm::ParameterSet retPSet;

  // example usages of HLTEgammaGenericFilter are:
  //   R9 shape filter                        hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolR9ShapeFilter 
  //   cluster shape filter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolClusterShapeFilter 
  //   Ecal isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TIghterEleIdIsolEcalIsolFilter
  //   H/E filter                             hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHEFilter
  //   HCAL isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHcalIsolFilter

  // the type of object to look for seems to be the
  // same for all uses of HLTEgammaGenericFilter
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);  

  // infer the type of filter by the type of the producer which
  // generates the collection used to cut on this
  edm::InputTag isoTag = hltConfig_.modulePSet(moduleName).getParameter<edm::InputTag>("isoTag");
  edm::InputTag nonIsoTag = hltConfig_.modulePSet(moduleName).getParameter<edm::InputTag>("nonIsoTag");
  //std::cout << "isoTag.label " << isoTag.label() << " nonIsoTag.label " << nonIsoTag.label() << std::endl;

  std::string inputType = hltConfig_.moduleType(isoTag.label());
  //std::cout << "inputType " << inputType << " moduleName " << moduleName << std::endl;

  //--------------------
  // sanity check: non-isolated path should be produced by the
  // same type of module

  // first check that the non-iso tag is non-empty
  if (nonIsoTag.label().empty()) {
    edm::LogError("EmDQMFeeder") << "nonIsoTag of HLTEgammaGenericFilter '" << moduleName <<  "' is empty.";
    throw "noNonIsoTag";
  }
  if (inputType != hltConfig_.moduleType(nonIsoTag.label())) {
    edm::LogError("EmDQMFeeder") << "C++ Type of isoTag '" << inputType << "' and nonIsoTag '" << hltConfig_.moduleType(nonIsoTag.label()) << "' are not the same for HLTEgammaGenericFilter '" << moduleName <<  "'.";
    throw "inputTypeNonMatching";
  }
  //--------------------

  std::vector<edm::InputTag> isoCollections;
  isoCollections.push_back(isoTag);
  isoCollections.push_back(nonIsoTag);

  //--------------------
  // the following cases seem to have identical PSets ?
  //--------------------

  //--------------------
  // R9 shape
  //--------------------
  if (inputType == "EgammaHLTR9Producer") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }
 
  //--------------------
  // R9 ID
  //--------------------
  if (inputType == "EgammaHLTR9IDProducer") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }

  //--------------------
  // cluster shape
  //--------------------
  if (inputType == "EgammaHLTClusterShapeProducer") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }

  //--------------------
  // ecal isolation
  //--------------------
  if (inputType == "EgammaHLTEcalRecIsolationProducer") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }

  //--------------------
  // HCAL isolation and HE
  //--------------------
  if (inputType == "EgammaHLTHcalIsolationProducersRegional") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }

   edm::LogError("EmDQMFeeder") << "Can't determine what the HLTEgammaGenericFilter '" << moduleName <<  "' should do: uses a collection produced by a module of C++ type '" << inputType << "'.";
   throw "unknownC++Type";

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForEgammaGenericQuadraticFilter(const std::string& pathName, const std::string& moduleName)
{
  edm::ParameterSet retPSet;

  // example usages of HLTEgammaGenericFilter are:
  //   R9 shape filter                        hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolR9ShapeFilter 
  //   cluster shape filter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolClusterShapeFilter 
  //   Ecal isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TIghterEleIdIsolEcalIsolFilter
  //   H/E filter                             hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHEFilter
  //   HCAL isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHcalIsolFilter

  // the type of object to look for seems to be the
  // same for all uses of HLTEgammaGenericFilter
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);  

  // infer the type of filter by the type of the producer which
  // generates the collection used to cut on this
  edm::InputTag isoTag = hltConfig_.modulePSet(moduleName).getParameter<edm::InputTag>("isoTag");
  edm::InputTag nonIsoTag = hltConfig_.modulePSet(moduleName).getParameter<edm::InputTag>("nonIsoTag");
  //std::cout << "isoTag.label " << isoTag.label() << " nonIsoTag.label " << nonIsoTag.label() << std::endl;

  std::string inputType = hltConfig_.moduleType(isoTag.label());
  //std::cout << "inputType " << inputType << " moduleName " << moduleName << std::endl;

  //--------------------
  // sanity check: non-isolated path should be produced by the
  // same type of module

  // first check that the non-iso tag is non-empty
  if (nonIsoTag.label().empty()) {
    edm::LogError("EmDQMFeeder") << "nonIsoTag of HLTEgammaGenericFilter '" << moduleName <<  "' is empty.";
    throw "noNonIsoTag";
  }
  if (inputType != hltConfig_.moduleType(nonIsoTag.label())) {
    edm::LogError("EmDQMFeeder") << "C++ Type of isoTag '" << inputType << "' and nonIsoTag '" << hltConfig_.moduleType(nonIsoTag.label()) << "' are not the same for HLTEgammaGenericFilter '" << moduleName <<  "'.";
    throw "inputTypeNonMatching";
  }
  //--------------------

  std::vector<edm::InputTag> isoCollections;
  isoCollections.push_back(isoTag);
  isoCollections.push_back(nonIsoTag);

  //--------------------
  // the following cases seem to have identical PSets ?
  //--------------------

  //--------------------
  // R9 shape
  //--------------------
  if (inputType == "EgammaHLTR9Producer") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }
 
  //--------------------
  // R9 ID
  //--------------------
  if (inputType == "EgammaHLTR9IDProducer") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }

  //--------------------
  // cluster shape
  //--------------------
  if (inputType == "EgammaHLTClusterShapeProducer") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }

  //--------------------
  // ecal isolation
  //--------------------
  if (inputType == "EgammaHLTEcalRecIsolationProducer") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }

  //--------------------
  // HCAL isolation and HE
  //--------------------
  if (inputType == "EgammaHLTHcalIsolationProducersRegional") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }

   edm::LogError("EmDQMFeeder") << "Can't determine what the HLTEgammaGenericQuadraticFilter '" << moduleName <<  "' should do: uses a collection produced by a module of C++ type '" << inputType << "'.";
   throw "unknownC++Type";

  return retPSet;
}


//----------------------------------------------------------------------

edm::ParameterSet
EmDQMFeeder::makePSetForElectronGenericFilter(const std::string& pathName, const std::string& moduleName)
{
  edm::ParameterSet retPSet;

  // example usages of HLTElectronGenericFilter are:
  //
  // deta filter      hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDetaFilter
  // dphi filter      hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDphiFilter
  // track isolation  hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolTrackIsolFilter
  //
  // the type of object to look for seems to be the
  // same for all uses of HLTEgammaGenericFilter
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerElectron);

  // infer the type of filter by the type of the producer which
  // generates the collection used to cut on this
  edm::InputTag isoTag = hltConfig_.modulePSet(moduleName).getParameter<edm::InputTag>("isoTag");
  edm::InputTag nonIsoTag = hltConfig_.modulePSet(moduleName).getParameter<edm::InputTag>("nonIsoTag");
  //std::cout << "isoTag.label " << isoTag.label() << " nonIsoTag.label " << nonIsoTag.label() << std::endl;

  std::string inputType = hltConfig_.moduleType(isoTag.label());
  //std::cout << "inputType iso " << inputType << " inputType noniso " << hltConfig_.moduleType(nonIsoTag.label()) << " moduleName " << moduleName << std::endl;

  //--------------------
  // sanity check: non-isolated path should be produced by the
  // same type of module
  if (nonIsoTag.label().empty()) {
    edm::LogError("EmDQMFeeder") << "nonIsoTag of HLTElectronGenericFilter '" << moduleName <<  "' is empty.";
    throw "noNonIsoTag";
  }
  if (inputType != hltConfig_.moduleType(nonIsoTag.label())) {
    edm::LogError("EmDQMFeeder") << "C++ Type of isoTag '" << inputType << "' and nonIsoTag '" << hltConfig_.moduleType(nonIsoTag.label()) << "' are not the same for HLTElectronGenericFilter '" << moduleName <<  "'.";
    throw "inputTypeNonMatching";
  }
  //--------------------

  std::vector<edm::InputTag> isoCollections;
  isoCollections.push_back(isoTag);
  isoCollections.push_back(nonIsoTag);

  //--------------------
  // the following cases seem to have identical PSets ?
  //--------------------

  //--------------------
  // deta and dphi filter
  //--------------------
  //
  // note that whether deta or dphi is used is determined from
  // the product instance (not the module label)
  if (inputType == "EgammaHLTElectronDetaDphiProducer") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }
 
  //--------------------
  // track isolation
  //--------------------
  if (inputType == "EgammaHLTElectronTrackIsolationProducers") {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    //retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

    return retPSet;
  }
 
   edm::LogError("EmDQMFeeder") << "Can't determine what the HLTElectronGenericFilter '" << moduleName <<  "' should do: uses a collection produced by a module of C++ type '" << inputType << "'.";
   throw "unknownC++Type";

  return retPSet;
}

//----------------------------------------------------------------------

DEFINE_FWK_MODULE(EmDQMFeeder);
