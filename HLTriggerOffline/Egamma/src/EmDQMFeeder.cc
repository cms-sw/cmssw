// -*- C++ -*-
//
// Package:    EmDQMFeeder
// Class:      EmDQMFeeder
// 
//
// Original Author:  Thomas Reis,40 4-B24,+41227671567,
//         Created:  Tue Mar 15 12:24:11 CET 2011
// $Id: EmDQMFeeder.cc,v 1.23 2012/03/17 20:43:14 treis Exp $
//
//

// class header file
#include "HLTriggerOffline/Egamma/interface/EmDQMFeeder.h"

//////////////////////////////////////////////////////////////////////////////
// constructors and destructor
//////////////////////////////////////////////////////////////////////////////
EmDQMFeeder::EmDQMFeeder(const edm::ParameterSet& iConfig_) :
  iConfig(iConfig_)
{
   //now do what ever initialization is needed
   //processName_ = iConfig_.getParameter<std::string>("processname");
   triggerObject_ = iConfig_.getParameter<edm::InputTag>("triggerobject");
   verbosity_ = iConfig_.getUntrackedParameter<unsigned int>("verbosity",0);
}


EmDQMFeeder::~EmDQMFeeder()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//////////////////////////////////////////////////////////////////////////////
// member functions
//////////////////////////////////////////////////////////////////////////////

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
//  for (unsigned i = 0; i < emDQMmodules.size(); ++i)
//     emDQMmodules[i]->beginJob();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EmDQMFeeder::endJob() 
{
  for (unsigned i = 0; i < emDQMmodules.size(); ++i)
     emDQMmodules[i]->endJob();
}

// ------------ method called when starting to processes a run  ------------
void 
EmDQMFeeder::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
   bool changed(true);
   //if (hltConfig_.init(iRun, iSetup, processName_, changed)) {
   if (hltConfig_.init(iRun, iSetup, triggerObject_.process(), changed)) {

      // if init returns TRUE, initialisation has succeeded!

      if (verbosity_ >= OUTPUT_ALL) {
         // Output general information on the menu
         edm::LogPrint("EmDQMFeeder") << "inited=" << hltConfig_.inited();
         edm::LogPrint("EmDQMFeeder") << "changed=" << hltConfig_.changed();
         edm::LogPrint("EmDQMFeeder") << "processName=" << hltConfig_.processName();
         edm::LogPrint("EmDQMFeeder") << "tableName=" << hltConfig_.tableName();
         edm::LogPrint("EmDQMFeeder") << "size=" << hltConfig_.size();
      }

      // All electron and photon paths
      std::vector<std::vector<std::string> > egammaPaths = findEgammaPaths();
      //std::cout << "Found " << egammaPaths[TYPE_SINGLE_ELE].size() << " single electron paths" << std::endl;
      //std::cout << "Found " << egammaPaths[TYPE_DOUBLE_ELE].size() << " double electron paths" << std::endl;
      //std::cout << "Found " << egammaPaths[TYPE_TRIPLE_ELE].size() << " triple electron paths" << std::endl;
      //std::cout << "Found " << egammaPaths[TYPE_SINGLE_PHOTON].size() << " single photon paths" << std::endl;
      //std::cout << "Found " << egammaPaths[TYPE_DOUBLE_PHOTON].size() << " double photon paths" << std::endl;

      std::vector<std::string> filterModules;

      for (unsigned int j=0; j < egammaPaths.size() ; j++) {

         for (unsigned int i=0; i < egammaPaths.at(j).size() ; i++) {
            // get pathname of this trigger 
	    const std::string pathName = egammaPaths.at(j).at(i);
            if (verbosity_ >= OUTPUT_ALL)
               edm::LogPrint("EmDQMFeeder") << "Path: " << pathName;

            // get filters of the current path
            filterModules = getFilterModules(pathName);

            //--------------------	   
	    edm::ParameterSet paramSet;

	    paramSet.addUntrackedParameter("pathIndex", hltConfig_.triggerIndex(pathName));
	    paramSet.addParameter("@module_label", hltConfig_.removeVersion(pathName) + "_DQM");
	    //paramSet.addParameter("@module_label", pathName + "_DQM");
	    //paramSet.addParameter("triggerobject", iConfig.getParameter<edm::InputTag>("triggerobject"));
	    paramSet.addParameter("triggerobject", triggerObject_);
	    paramSet.addParameter("genEtaAcc", iConfig.getParameter<double>("genEtaAcc"));
	    paramSet.addParameter("genEtAcc", iConfig.getParameter<double>("genEtAcc"));

	    // plotting parameters (untracked because they don't affect the physics)
            double genEtMin = getPrimaryEtCut(pathName);
            if (genEtMin >= 0) {
               paramSet.addUntrackedParameter("genEtMin", genEtMin);
            } else {
               if (verbosity_ >= OUTPUT_WARNINGS)
                  edm::LogWarning("EmDQMFeeder") << "Pathname: '" << pathName << "':  Unable to determine a minimum Et. Will not include this path in the validation.";
               continue;
            }

            // set the x axis of the et plots to some reasonable value based
            // on the primary et cut determined from the path name
            double ptMax = iConfig.getUntrackedParameter<double>("PtMax",1000.);
            double ptMin = iConfig.getUntrackedParameter<double>("PtMin",0.);
            if (ptMax < (1.2*genEtMin)) {
               paramSet.addUntrackedParameter<double>("PtMax", (10*ceil(0.12 * genEtMin)));
               paramSet.addUntrackedParameter<double>("PtMin", (10*ceil(0.12 * genEtMin) - ptMax + ptMin));
            }
	    else {
               paramSet.addUntrackedParameter<double>("PtMax", ptMax);
               paramSet.addUntrackedParameter<double>("PtMin", ptMin);
            }

	    paramSet.addUntrackedParameter("EtaMax", iConfig.getUntrackedParameter<double>("EtaMax", 2.7));
	    paramSet.addUntrackedParameter("PhiMax", iConfig.getUntrackedParameter<double>("PhiMax", 3.15));
	    paramSet.addUntrackedParameter("Nbins", iConfig.getUntrackedParameter<unsigned int>("Nbins",40));
	    paramSet.addUntrackedParameter("minEtForEtaEffPlot", iConfig.getUntrackedParameter<unsigned int>("minEtForEtaEffPlot", 15));
	    paramSet.addUntrackedParameter("useHumanReadableHistTitles", iConfig.getUntrackedParameter<bool>("useHumanReadableHistTitles", false));
            paramSet.addUntrackedParameter("mcMatchedOnly", iConfig.getUntrackedParameter<bool>("mcMatchedOnly", true));
            paramSet.addUntrackedParameter("noPhiPlots", iConfig.getUntrackedParameter<bool>("noPhiPlots", true));
            paramSet.addUntrackedParameter("noIsolationPlots", iConfig.getUntrackedParameter<bool>("noIsolationPlots", true));

            // verbosity of output messages when running
            paramSet.addUntrackedParameter<unsigned int>("verbosity", verbosity_);

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
               case TYPE_TRIPLE_ELE:
 	          paramSet.addParameter<unsigned>("reqNum", 3);
	          paramSet.addParameter<int>("pdgGen", 11);
                  paramSet.addParameter<edm::InputTag>("cutcollection", edm::InputTag("fiducialZee"));
	          paramSet.addParameter<int>("cutnum", 3);
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

            // TODO: extend this
            if (iConfig.getParameter<bool>("isData")) paramSet.addParameter<edm::InputTag>("cutcollection", edm::InputTag("gsfElectrons"));

            std::vector<edm::ParameterSet> filterVPSet;
            edm::ParameterSet filterPSet;
            std::string moduleLabel;

            // loop over filtermodules of current trigger path
            for (std::vector<std::string>::iterator filter = filterModules.begin(); filter != filterModules.end(); ++filter) {
               std::string moduleType = hltConfig_.modulePSet(*filter).getParameter<std::string>("@module_type");
               moduleLabel = hltConfig_.modulePSet(*filter).getParameter<std::string>("@module_label");

               // first check if it is a filter we are not interrested in
               if (moduleType == "Pythia6GeneratorFilter" ||
                   moduleType == "HLTTriggerTypeFilter" ||
                   moduleType == "HLTLevel1Activity" ||
                   moduleType == "HLTPrescaler" ||
                   moduleType == "HLTBool")
                  continue;

               // now check for the known filter types
               if (moduleType == "HLTLevel1GTSeed") {
                  filterPSet = makePSetForL1SeedFilter(moduleLabel);
               }
               else if (moduleType == "HLTEgammaL1MatchFilterRegional") {
                  filterPSet = makePSetForL1SeedToSuperClusterMatchFilter(moduleLabel);
               }
               else if (moduleType == "HLTEgammaEtFilter") {
                  filterPSet = makePSetForEtFilter(moduleLabel);
               }
               else if (moduleType == "HLTElectronOneOEMinusOneOPFilterRegional") {
                  filterPSet = makePSetForOneOEMinusOneOPFilter(moduleLabel);
               }
               else if (moduleType == "HLTElectronPixelMatchFilter") {
                  filterPSet = makePSetForPixelMatchFilter(moduleLabel);
               }
               else if (moduleType == "HLTEgammaGenericFilter") {
                  filterPSet = makePSetForEgammaGenericFilter(moduleLabel);
               }
               else if (moduleType == "HLTEgammaGenericQuadraticFilter") {
                  filterPSet = makePSetForEgammaGenericQuadraticFilter(moduleLabel);
               }
               else if (moduleType == "HLTElectronGenericFilter") {
                  filterPSet = makePSetForElectronGenericFilter(moduleLabel);
               }
               else if (moduleType == "HLTEgammaDoubleEtDeltaPhiFilter") {
                  filterPSet = makePSetForEgammaDoubleEtDeltaPhiFilter(moduleLabel);
               }
               else if (moduleType == "HLTGlobalSumsMET"
                        || moduleType == "HLTMhtHtFilter"
                        || moduleType == "HLTJetTag"
                        || moduleType == "HLT1CaloJet"
                        || moduleType == "HLT1CaloBJet"
                        || moduleType == "HLT1Tau"
                        || moduleType == "PFTauSelector"
                        || moduleType == "EtMinCaloJetSelector"
                        || moduleType == "LargestEtCaloJetSelector"
                        || moduleType == "HLTEgammaTriggerFilterObjectWrapper"  // 'fake' filter
                        || moduleType == "HLTEgammaDoubleLegCombFilter" // filter does not put anything in TriggerEventWithRefs
                        //|| moduleType == "HLT2ElectronTau"
                        || moduleType == "HLTPMMassFilter"
                        || moduleType == "HLTHcalTowerFilter"
                        //|| moduleType == "HLT1Photon"
                       )
                  continue;
               else {
                  if (verbosity_ >= OUTPUT_WARNINGS)
                     edm::LogWarning("EmDQMFeeder")  << "No parameter set for filter '" << moduleLabel << "' with filter type '" << moduleType << "' added. Module will not be analyzed.";
                  continue;
               }

               // something went wrong when the parameter set is empty. 
               if (!filterPSet.empty()) {
                  if (!hltConfig_.modulePSet(moduleLabel).exists("saveTags")) {
                     // make sure that 'theHLTOutputTypes' before an unseeded filter in a photon path is set to trigger::TriggerPhoton
                     // this is coupled to the parameter 'saveTag = true'
                     if (moduleLabel.find("Unseeded") != std::string::npos && (j == TYPE_DOUBLE_PHOTON || j == TYPE_SINGLE_PHOTON)) {
                        filterVPSet.back().addParameter<int>("theHLTOutputTypes", trigger::TriggerPhoton);
                     }
                  }
                  // if ncandcut is -1 (when parsing for the number of particles in the name of the L1seed filter fails),
                  // fall back to setting ncandcut to the number of particles needed for the given path.
                  if (filterPSet.getParameter<int>("ncandcut") < 0) filterPSet.addParameter<int>("ncandcut", paramSet.getParameter<int>("cutnum"));
                  else if (filterPSet.getParameter<int>("ncandcut") > paramSet.getParameter<int>("cutnum")) {
                    paramSet.addParameter<int>("cutnum", filterPSet.getParameter<int>("ncandcut"));
                    paramSet.addParameter<unsigned>("reqNum", (unsigned)filterPSet.getParameter<int>("ncandcut"));
                  }

                  filterVPSet.push_back(filterPSet);
               }
               else
                  break;

            } // end loop over filter modules of current trigger path

            // do not include this path when an empty filterPSet is detected.
            if (!filterPSet.empty()) {
               std::string lastModuleName = filterPSet.getParameter<edm::InputTag>("HLTCollectionLabels").label();
               if (!hltConfig_.modulePSet(lastModuleName).exists("saveTags")) {
                  // make sure that 'theHLTOutputTypes' of the last filter of a photon path is set to trigger::TriggerPhoton
                  // this is coupled to the parameter 'saveTag = true'
                  if ((j == TYPE_SINGLE_PHOTON || j == TYPE_DOUBLE_PHOTON) && pathName.rfind("Ele") == std::string::npos) {
                     filterVPSet.back().addParameter<int>("theHLTOutputTypes", trigger::TriggerPhoton);
                  }
               }
               paramSet.addParameter<std::vector<edm::ParameterSet> >("filters", filterVPSet);
            }
            else {
               if (verbosity_ >= OUTPUT_ALL)
                  edm::LogPrint("EmDQMFeeder") << "Will not include this path in the validation due to errors while generating the parameter set.";
               continue;
            }

            // dump generated parameter set
            //std::cout << paramSet.dump() << std::endl;

	    emDQMmodules.push_back(new EmDQM(paramSet));

	    emDQMmodules.back()->beginJob();
	    emDQMmodules.back()->beginRun(iRun, iSetup);
         } // loop over all paths of this analysis type

      } // loop over analysis types (single ele etc.)

      if (changed) {
         // The HLT config has actually changed wrt the previous Run, hence rebook your
         // histograms or do anything else dependent on the revised HLT config
      }
   } else {
      // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
      // with the file and/or code and needs to be investigated!
      if (verbosity_ >= OUTPUT_ERRORS)
         edm::LogError("EmDQMFeeder") << " HLT config extraction failure with process name '" << triggerObject_.process() << "'.";
      // In this case, all access methods will return empty values!
   }
}

// ------------ method called when ending the processing of a run  ------------
void 
EmDQMFeeder::endRun(edm::Run const&iEvent, edm::EventSetup const&iSetup)
{
  for (unsigned i = 0; i < emDQMmodules.size(); ++i)
     emDQMmodules[i]->endRun(iEvent, iSetup);
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
   std::vector<std::vector<std::string> > Paths(5);
   // Loop over all paths in the menu
   for (unsigned int i=0; i<hltConfig_.size(); i++) {

      std::string path = hltConfig_.triggerName(i);

      // Find electron and photon paths
      if (int(path.find("HLT_")) == 0) {    // Path should start with 'HLT_'
         if (path.find("HLT_Ele") != std::string::npos && path.rfind("Ele") == 4 && path.find("SC") == std::string::npos) {
            Paths[TYPE_SINGLE_ELE].push_back(path);
            //std::cout << "Electron " << path << std::endl;
         }
         if (path.find("HLT_Ele") != std::string::npos && path.find("EleId") != std::string::npos && path.rfind("Ele") == path.find("EleId")) {
            Paths[TYPE_SINGLE_ELE].push_back(path);
            //std::cout << "Electron " << path << std::endl;
         }
         else if (path.find("HLT_Ele") != std::string::npos && path.rfind("Ele") > 4) {
            Paths[TYPE_DOUBLE_ELE].push_back(path);
            //std::cout << "DoubleElectron " << path << std::endl;
         }
         else if (path.find("HLT_DoubleEle") != std::string::npos && path.find("Ele") == path.rfind("Ele")) {
            Paths[TYPE_DOUBLE_ELE].push_back(path);
            //std::cout << "DoubleElectron " << path << std::endl;
         }
         else if (path.find("HLT_Ele") != std::string::npos && path.find("SC") != std::string::npos) {
            Paths[TYPE_DOUBLE_ELE].push_back(path);
            //std::cout << "DoubleElectron " << path << std::endl;
         }
         else if (path.find("HLT_DoubleEle") != std::string::npos && path.find("Ele") != path.rfind("Ele")) {
            Paths[TYPE_TRIPLE_ELE].push_back(path);
            //std::cout << "TripleElectron " << path << std::endl;
         }
         else if (path.find("HLT_TripleEle") != std::string::npos && path.find("Ele") == path.rfind("Ele")) {
            Paths[TYPE_TRIPLE_ELE].push_back(path);
            //std::cout << "TripleElectron " << path << std::endl;
         }
         else if (path.find("HLT_Photon") != std::string::npos && path.find("Ele") != std::string::npos) {
            Paths[TYPE_DOUBLE_PHOTON].push_back(path);
            //std::cout << "DoublePhoton " << path << std::endl;
         }
         else if (path.find("HLT_Photon") != std::string::npos && path.rfind("Photon") == 4) {
            Paths[TYPE_SINGLE_PHOTON].push_back(path);
            //std::cout << "Photon " << path << std::endl;
         }
         else if (path.find("HLT_Photon") != std::string::npos && path.rfind("Photon") > 4) {
            Paths[TYPE_DOUBLE_PHOTON].push_back(path);
            //std::cout << "DoublePhoton " << path << std::endl;
         }
         else if (path.find("HLT_DoublePhoton") != std::string::npos) {
            Paths[TYPE_DOUBLE_PHOTON].push_back(path);
            //std::cout << "DoublePhoton " << path << std::endl;
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
  //retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", triggerObject_.process()));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerL1NoIsoEG);
  
  // as HLTLevel1GTSeed has no parameter ncandcut we determine the value from the name of the filter
  if (moduleName.find("Single") != std::string::npos)
     retPSet.addParameter<int>("ncandcut", 1);
  else if (moduleName.find("Double") != std::string::npos)
     retPSet.addParameter<int>("ncandcut", 2);
  else if (moduleName.find("Triple") != std::string::npos)
     retPSet.addParameter<int>("ncandcut", 3);
  else
     retPSet.addParameter<int>("ncandcut", -1);

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
  //retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", triggerObject_.process()));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);
  retPSet.addParameter<int>("ncandcut", hltConfig_.modulePSet(moduleName).getParameter<int>("ncandcut"));

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForEtFilter(const std::string& moduleName)
{
  edm::ParameterSet retPSet;
  
  retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
  //retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", triggerObject_.process()));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);
  retPSet.addParameter<int>("ncandcut", hltConfig_.modulePSet(moduleName).getParameter<int>("ncandcut"));

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForOneOEMinusOneOPFilter(const std::string& moduleName)
{
  edm::ParameterSet retPSet;
  
  retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
  //retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", triggerObject_.process()));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerElectron);
  retPSet.addParameter<int>("ncandcut", hltConfig_.modulePSet(moduleName).getParameter<int>("ncandcut"));

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForPixelMatchFilter(const std::string& moduleName)
{
  edm::ParameterSet retPSet;
 
  retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
  //retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", triggerObject_.process()));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);
  retPSet.addParameter<int>("ncandcut", hltConfig_.modulePSet(moduleName).getParameter<int>("ncandcut"));

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForEgammaDoubleEtDeltaPhiFilter(const std::string& moduleName)
{
  edm::ParameterSet retPSet;
 
  retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
  //retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
  retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", triggerObject_.process()));
  retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", std::vector<edm::InputTag>(1, std::string("none")));
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);
  retPSet.addParameter<int>("ncandcut", 2);

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForEgammaGenericFilter(const std::string& moduleName)
{
  edm::ParameterSet retPSet;

  // example usages of HLTEgammaGenericFilter are:
  //   R9 shape filter                        hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolR9ShapeFilter 
  //   cluster shape filter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolClusterShapeFilter 
  //   Ecal isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TIghterEleIdIsolEcalIsolFilter
  //   H/E filter                             hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHEFilter
  //   HCAL isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHcalIsolFilter

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
  //if (nonIsoTag.label().empty()) {
  //  edm::LogError("EmDQMFeeder") << "nonIsoTag of HLTEgammaGenericFilter '" << moduleName <<  "' is empty.";
  //  return retPSet;
  //}
  //if (inputType != hltConfig_.moduleType(nonIsoTag.label())) {
  //  edm::LogError("EmDQMFeeder") << "C++ Type of isoTag '" << inputType << "' and nonIsoTag '" << hltConfig_.moduleType(nonIsoTag.label()) << "' are not the same for HLTEgammaGenericFilter '" << moduleName <<  "'.";
  //  return retPSet;
  //}
  //--------------------

  // parameter saveTag determines the output type
  if (hltConfig_.saveTags(moduleName))
     retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerPhoton);  
  else
     retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

  std::vector<edm::InputTag> isoCollections;
  isoCollections.push_back(isoTag);
  if (!nonIsoTag.label().empty())
     isoCollections.push_back(nonIsoTag);

  //--------------------
  // the following cases seem to have identical PSets ?
  //--------------------

  if (inputType == "EgammaHLTR9Producer" ||                       // R9 shape
      inputType == "EgammaHLTR9IDProducer" ||                     // R9 ID
      inputType == "EgammaHLTClusterShapeProducer" ||             // cluster shape
      inputType == "EgammaHLTEcalRecIsolationProducer" ||         // ecal isolation
      inputType == "EgammaHLTHcalIsolationProducersRegional" ||   // HCAL isolation and HE
      inputType == "EgammaHLTGsfTrackVarProducer"                 // GSF track deta and dphi filter
     ) {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    //retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", triggerObject_.process()));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    retPSet.addParameter<int>("ncandcut", hltConfig_.modulePSet(moduleName).getParameter<int>("ncandcut"));

    return retPSet;
  }

  if (verbosity_ >= OUTPUT_ERRORS)
     edm::LogError("EmDQMFeeder") << "Can't determine what the HLTEgammaGenericFilter '" << moduleName <<  "' should do: uses a collection produced by a module of C++ type '" << inputType << "'.";
  return edm::ParameterSet();
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQMFeeder::makePSetForEgammaGenericQuadraticFilter(const std::string& moduleName)
{
  edm::ParameterSet retPSet;

  // example usages of HLTEgammaGenericFilter are:
  //   R9 shape filter                        hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolR9ShapeFilter 
  //   cluster shape filter                   hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolClusterShapeFilter 
  //   Ecal isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TIghterEleIdIsolEcalIsolFilter
  //   H/E filter                             hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHEFilter
  //   HCAL isolation filter                  hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolHcalIsolFilter

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
  //if (nonIsoTag.label().empty()) {
  //  edm::LogError("EmDQMFeeder") << "nonIsoTag of HLTEgammaGenericFilter '" << moduleName <<  "' is empty.";
  //  return retPSet;
  //}
  //if (inputType != hltConfig_.moduleType(nonIsoTag.label())) {
  //  edm::LogError("EmDQMFeeder") << "C++ Type of isoTag '" << inputType << "' and nonIsoTag '" << hltConfig_.moduleType(nonIsoTag.label()) << "' are not the same for HLTEgammaGenericFilter '" << moduleName <<  "'.";
  //  return retPSet;
  //}
  //--------------------

  // parameter saveTag determines the output type
  if (hltConfig_.saveTags(moduleName))
     retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerPhoton);  
  else
     retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerCluster);

  std::vector<edm::InputTag> isoCollections;
  isoCollections.push_back(isoTag);
  if (!nonIsoTag.label().empty())
     isoCollections.push_back(nonIsoTag);

  //--------------------
  // the following cases seem to have identical PSets ?
  //--------------------

  if (inputType == "EgammaHLTR9Producer" ||                            // R9 shape
      inputType == "EgammaHLTR9IDProducer" ||                          // R9 ID
      inputType == "EgammaHLTClusterShapeProducer" ||                  // cluster shape
      inputType == "EgammaHLTEcalRecIsolationProducer" ||              // ecal isolation
      inputType == "EgammaHLTHcalIsolationProducersRegional" ||        // HCAL isolation and HE
      inputType == "EgammaHLTPhotonTrackIsolationProducersRegional"    // Photon track isolation
     ) {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    //retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", triggerObject_.process()));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    retPSet.addParameter<int>("ncandcut", hltConfig_.modulePSet(moduleName).getParameter<int>("ncandcut"));

    return retPSet;
  }
 
  if (verbosity_ >= OUTPUT_ERRORS)
     edm::LogError("EmDQMFeeder") << "Can't determine what the HLTEgammaGenericQuadraticFilter '" << moduleName <<  "' should do: uses a collection produced by a module of C++ type '" << inputType << "'.";
  return edm::ParameterSet();
}


//----------------------------------------------------------------------

edm::ParameterSet
EmDQMFeeder::makePSetForElectronGenericFilter(const std::string& moduleName)
{
  edm::ParameterSet retPSet;

  // example usages of HLTElectronGenericFilter are:
  //
  // deta filter      hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDetaFilter
  // dphi filter      hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolDphiFilter
  // track isolation  hltL1NonIsoHLTNonIsoSingleElectronEt17TighterEleIdIsolTrackIsolFilter

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
  //if (nonIsoTag.label().empty()) {
  //  edm::LogError("EmDQMFeeder") << "nonIsoTag of HLTElectronGenericFilter '" << moduleName <<  "' is empty.";
  //  return retPSet;
  //}
  //if (inputType != hltConfig_.moduleType(nonIsoTag.label())) {
  //  edm::LogError("EmDQMFeeder") << "C++ Type of isoTag '" << inputType << "' and nonIsoTag '" << hltConfig_.moduleType(nonIsoTag.label()) << "' are not the same for HLTElectronGenericFilter '" << moduleName <<  "'.";
  //  return retPSet;
  //}
  //--------------------

  // the type of object to look for seems to be the
  // same for all uses of HLTEgammaGenericFilter
  retPSet.addParameter<int>("theHLTOutputTypes", trigger::TriggerElectron);

  std::vector<edm::InputTag> isoCollections;
  isoCollections.push_back(isoTag);
  if (!nonIsoTag.label().empty())
     isoCollections.push_back(nonIsoTag);

  //--------------------
  // the following cases seem to have identical PSets ?
  //--------------------

  // note that whether deta or dphi is used is determined from
  // the product instance (not the module label)
  if (inputType == "EgammaHLTElectronDetaDphiProducer" ||           // deta and dphi filter
      inputType == "EgammaHLTElectronTrackIsolationProducers"       // track isolation
     ) {
    retPSet.addParameter<std::vector<double> >("PlotBounds", std::vector<double>(2, 0.0));
    //retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", processName_));
    retPSet.addParameter<edm::InputTag>("HLTCollectionLabels", edm::InputTag(moduleName, "", triggerObject_.process()));
    retPSet.addParameter<std::vector<edm::InputTag> >("IsoCollections", isoCollections);
    retPSet.addParameter<int>("ncandcut", hltConfig_.modulePSet(moduleName).getParameter<int>("ncandcut"));

    return retPSet;
  }

  if (verbosity_ >= OUTPUT_ERRORS)
     edm::LogError("EmDQMFeeder") << "Can't determine what the HLTElectronGenericFilter '" << moduleName <<  "' should do: uses a collection produced by a module of C++ type '" << inputType << "'.";
  return edm::ParameterSet();
}

//----------------------------------------------------------------------

DEFINE_FWK_MODULE(EmDQMFeeder);
