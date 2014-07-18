///////////////////////////////////////////////////////////////////////////////
//                    Header file for this                                    //
////////////////////////////////////////////////////////////////////////////////
#include "HLTriggerOffline/Egamma/interface/EmDQM.h"

using namespace ROOT::Math::VectorUtil ;


////////////////////////////////////////////////////////////////////////////////
//                             Constructor                                    //
////////////////////////////////////////////////////////////////////////////////
EmDQM::EmDQM(const edm::ParameterSet& pset_) : pset(pset_) 
{
  // are we running in automatic configuration mode with the HLTConfigProvider
  // or with a per path python config file
  autoConfMode_ = pset.getUntrackedParameter<bool>("autoConfMode", false);

  // set global parameters
  triggerObject_ = pset_.getParameter<edm::InputTag>("triggerobject");
  verbosity_ = pset_.getUntrackedParameter<unsigned int>("verbosity",0);
  genEtaAcc_ = pset.getParameter<double>("genEtaAcc");
  genEtAcc_ = pset.getParameter<double>("genEtAcc");
  ptMax_ = pset.getUntrackedParameter<double>("PtMax",1000.);
  ptMin_ = pset.getUntrackedParameter<double>("PtMin",0.);
  etaMax_ = pset.getUntrackedParameter<double>("EtaMax", 2.7);
  phiMax_ = pset.getUntrackedParameter<double>("PhiMax", 3.15);
  nbins_ = pset.getUntrackedParameter<unsigned int>("Nbins",40);
  minEtForEtaEffPlot_ = pset.getUntrackedParameter<unsigned int>("minEtForEtaEffPlot", 15);
  useHumanReadableHistTitles_ = pset.getUntrackedParameter<bool>("useHumanReadableHistTitles", false);
  mcMatchedOnly_ = pset.getUntrackedParameter<bool>("mcMatchedOnly", true);
  noPhiPlots_ = pset.getUntrackedParameter<bool>("noPhiPlots", true);
  noIsolationPlots_ = pset.getUntrackedParameter<bool>("noIsolationPlots", true);

  if (!autoConfMode_) {
    paramSets.push_back(pset);
    isData_ = false;
  } else {
    isData_ = pset.getParameter<bool>("isData");
  }

  histoFillerEle      = new HistoFiller<reco::ElectronCollection>(this);
  histoFillerPho      = new HistoFiller<reco::RecoEcalCandidateCollection>(this);
  histoFillerClu      = new HistoFiller<reco::RecoEcalCandidateCollection>(this);
  histoFillerL1Iso    = new HistoFiller<l1extra::L1EmParticleCollection>(this);
  histoFillerL1NonIso = new HistoFiller<l1extra::L1EmParticleCollection>(this);

  // consumes
  genParticles_token = consumes<edm::View<reco::Candidate> >(edm::InputTag("genParticles", "", "SIM"));
  triggerObject_token = consumes<trigger::TriggerEventWithRefs>(triggerObject_);
  hltResults_token = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults", "", triggerObject_.process()));
  if (autoConfMode_) {
    gencutColl_fidWenu_token = mayConsume<edm::View<reco::Candidate> >(edm::InputTag("fiducialWenu"));
    gencutColl_fidZee_token = mayConsume<edm::View<reco::Candidate> >(edm::InputTag("fiducialZee"));
    gencutColl_fidTripleEle_token = mayConsume<edm::View<reco::Candidate> >(edm::InputTag("fiducialTripleEle"));
    gencutColl_fidGammaJet_token = mayConsume<edm::View<reco::Candidate> >(edm::InputTag("fiducialGammaJet"));
    gencutColl_fidDiGamma_token = mayConsume<edm::View<reco::Candidate> >(edm::InputTag("fiducialDiGamma"));
  } else {
    gencutColl_manualConf_token = consumes<edm::View<reco::Candidate> >(pset.getParameter<edm::InputTag>("cutcollection"));
  }
}

////////////////////////////////////////////////////////////////////////////////
//       method called once each job just before starting event loop          //
////////////////////////////////////////////////////////////////////////////////
void 
EmDQM::beginJob()
{

}

void 
EmDQM::dqmBeginRun(edm::Run const &iRun, edm::EventSetup const &iSetup)
{
   bool changed(true);
   if (hltConfig_.init(iRun, iSetup, triggerObject_.process(), changed)) {

      // if init returns TRUE, initialisation has succeeded!

      if (autoConfMode_) {
        if (verbosity_ >= OUTPUT_ALL) {
           // Output general information on the menu
           edm::LogPrint("EmDQM") << "inited=" << hltConfig_.inited();
           edm::LogPrint("EmDQM") << "changed=" << hltConfig_.changed();
           edm::LogPrint("EmDQM") << "processName=" << hltConfig_.processName();
           edm::LogPrint("EmDQM") << "tableName=" << hltConfig_.tableName();
           edm::LogPrint("EmDQM") << "size=" << hltConfig_.size();
           edm::LogInfo("EmDQM") << "The following filter types are not analyzed: \n" 
                                  << "\tHLTGlobalSumsMET\n"
                                  << "\tHLTHtMhtFilter\n"
                                  << "\tHLTMhtFilter\n"
                                  << "\tHLTJetTag\n"
                                  << "\tHLT1CaloJet\n"
                                  << "\tHLT1CaloMET\n"
                                  << "\tHLT1CaloBJet\n"
                                  << "\tHLT1Tau\n"
                                  << "\tHLT1PFTau\n"
                                  << "\tPFTauSelector\n"
                                  << "\tHLT1PFJet\n"
                                  << "\tHLTPFJetCollectionsFilter\n"
                                  << "\tHLTPFJetCollectionsVBFFilter\n"
                                  << "\tHLTPFJetTag\n"
                                  << "\tEtMinCaloJetSelector\n"
                                  << "\tEtMinPFJetSelector\n"
                                  << "\tLargestEtCaloJetSelector\n"
                                  << "\tLargestEtPFJetSelector\n"
                                  << "\tHLTEgammaTriggerFilterObjectWrapper\n"
                                  << "\tHLTEgammaDoubleLegCombFilter\n"
                                  << "\tHLT2ElectronTau\n"
                                  << "\tHLT2ElectronMET\n"
                                  << "\tHLT2ElectronPFTau\n"
                                  << "\tHLTPMMassFilter\n"
                                  << "\tHLTHcalTowerFilter\n"
                                  << "\tHLT1Photon\n"
                                  << "\tHLTRFilter\n"
                                  << "\tHLTRHemisphere\n"
                                  << "\tHLTElectronPFMTFilter\n"
                                  << "\tPrimaryVertexObjectFilter\n"
                                  << "\tHLTEgammaAllCombMassFilter\n"
                                  << "\tHLTMuon*\n"
                                  ;
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
           if (verbosity_ >= OUTPUT_ALL) {
              switch(j) {
                 case TYPE_SINGLE_ELE:
                    edm::LogPrint("EmDQM") << "/////////////////////////////////////////\nSingle electron paths: ";
                    break;
                 case TYPE_DOUBLE_ELE:
                    edm::LogPrint("EmDQM") << "/////////////////////////////////////////\nDouble electron paths: ";
                    break;
                 case TYPE_TRIPLE_ELE:
                    edm::LogPrint("EmDQM") << "/////////////////////////////////////////\nTriple electron paths: ";
                    break;
                 case TYPE_SINGLE_PHOTON:
                    edm::LogPrint("EmDQM") << "/////////////////////////////////////////\nSingle photon paths: ";
                    break;
                 case TYPE_DOUBLE_PHOTON:
                    edm::LogPrint("EmDQM") << "/////////////////////////////////////////\nDouble photon paths: ";
                    break;
              }
           }

           for (unsigned int i=0; i < egammaPaths.at(j).size() ; i++) {
              // get pathname of this trigger 
              const std::string pathName = egammaPaths.at(j).at(i);
              if (verbosity_ >= OUTPUT_ALL)
                 edm::LogPrint("EmDQM") << "Path: " << pathName;

              // get filters of the current path
              filterModules = getFilterModules(pathName);

              //--------------------	   
              edm::ParameterSet paramSet;

              paramSet.addUntrackedParameter("pathIndex", hltConfig_.triggerIndex(pathName));
              paramSet.addParameter("@module_label", hltConfig_.removeVersion(pathName) + "_DQM");
              //paramSet.addParameter("@module_label", pathName + "_DQM");

              // plotting parameters (untracked because they don't affect the physics)
              double genEtMin = getPrimaryEtCut(pathName);
              if (genEtMin >= 0) {
                 paramSet.addUntrackedParameter("genEtMin", genEtMin);
              } else {
                 if (verbosity_ >= OUTPUT_WARNINGS)
                    edm::LogWarning("EmDQM") << "Pathname: '" << pathName << "':  Unable to determine a minimum Et. Will not include this path in the validation.";
                 continue;
              }

              // set the x axis of the et plots to some reasonable value based
              // on the primary et cut determined from the path name
              double ptMax = ptMax_;
              double ptMin = ptMin_;
              if (ptMax < (1.2*genEtMin)) {
                 paramSet.addUntrackedParameter<double>("PtMax", (10*ceil(0.12 * genEtMin)));
                 paramSet.addUntrackedParameter<double>("PtMin", (10*ceil(0.12 * genEtMin) - ptMax + ptMin));
              }
              else {
                 paramSet.addUntrackedParameter<double>("PtMax", ptMax);
                 paramSet.addUntrackedParameter<double>("PtMin", ptMin);
              }

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
                    paramSet.addParameter<edm::InputTag>("cutcollection", edm::InputTag("fiducialTripleEle"));
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
              if (isData_) paramSet.addParameter<edm::InputTag>("cutcollection", edm::InputTag("gsfElectrons"));

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
                          || moduleType == "HLTHtMhtFilter"
                          || moduleType == "HLTMhtFilter"
                          || moduleType == "HLTJetTag"
                          || moduleType == "HLT1CaloJet"
                          || moduleType == "HLT1CaloMET"
                          || moduleType == "HLT1CaloBJet"
                          || moduleType == "HLT1Tau"
                          || moduleType == "HLT1PFTau"
                          || moduleType == "PFTauSelector"
                          || moduleType == "HLT1PFJet"
                          || moduleType == "HLTPFJetCollectionsFilter"
                          || moduleType == "HLTPFJetCollectionsVBFFilter"
                          || moduleType == "HLTPFJetTag"
                          || moduleType == "EtMinCaloJetSelector"
                          || moduleType == "EtMinPFJetSelector"
                          || moduleType == "LargestEtCaloJetSelector"
                          || moduleType == "LargestEtPFJetSelector"
                          || moduleType == "HLTEgammaTriggerFilterObjectWrapper"  // 'fake' filter
                          || moduleType == "HLTEgammaDoubleLegCombFilter" // filter does not put anything in TriggerEventWithRefs
                          || moduleType == "HLT2ElectronMET"
                          || moduleType == "HLT2ElectronTau"
                          || moduleType == "HLT2ElectronPFTau"
                          || moduleType == "HLTPMMassFilter"
                          || moduleType == "HLTHcalTowerFilter"
                          || moduleType == "HLT1Photon"
                          || moduleType == "HLTRFilter"
                          || moduleType == "HLTRHemisphere"
                          || moduleType == "HLTElectronPFMTFilter"
                          || moduleType == "PrimaryVertexObjectFilter"
                          || moduleType == "HLTEgammaAllCombMassFilter"
                          || moduleType.find("HLTMuon") != std::string::npos
                         )
                    continue;
                 else {
                    if (verbosity_ >= OUTPUT_WARNINGS)
                       edm::LogWarning("EmDQM")  << "No parameter set for filter '" << moduleLabel << "' with filter type '" << moduleType << "' added. Module will not be analyzed.";
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
                    edm::LogPrint("EmDQM") << "Will not include this path in the validation due to errors while generating the parameter set.";
                 continue;
              }

              // dump generated parameter set
              //std::cout << paramSet.dump() << std::endl;

              paramSets.push_back(paramSet);
           } // loop over all paths of this analysis type

        } // loop over analysis types (single ele etc.)
      }

      hltCollectionLabelsFoundPerPath.reserve(paramSets.size());
      hltCollectionLabelsMissedPerPath.reserve(paramSets.size());

      ////////////////////////////////////////////////////////////
      // loop over all the trigger path parameter sets
      ////////////////////////////////////////////////////////////
      for (std::vector<edm::ParameterSet>::iterator psetIt = paramSets.begin(); psetIt != paramSets.end(); ++psetIt) {
         hltCollectionLabelsFoundPerPath.push_back(hltCollectionLabelsFound);
         hltCollectionLabelsMissedPerPath.push_back(hltCollectionLabelsMissed);
      }

      if (changed) {
         // The HLT config has actually changed wrt the previous Run, hence rebook your
         // histograms or do anything else dependent on the revised HLT config
      }
   } else {
      // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
      // with the file and/or code and needs to be investigated!
      if (verbosity_ >= OUTPUT_ERRORS)
         edm::LogError("EmDQM") << " HLT config extraction failure with process name '" << triggerObject_.process() << "'.";
      // In this case, all access methods will return empty values!
   }
}

void 
EmDQM::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &iRun, edm::EventSetup const &iSetup)
{
   ////////////////////////////////////////////////////////////
   // loop over all the trigger path parameter sets
   ////////////////////////////////////////////////////////////
   for (std::vector<edm::ParameterSet>::iterator psetIt = paramSets.begin(); psetIt != paramSets.end(); ++psetIt) {
      SetVarsFromPSet(psetIt);

      iBooker.setCurrentFolder(dirname_);
    
      ////////////////////////////////////////////////////////////
      //  Set up Histogram of Effiency vs Step.                 //
      //   theHLTCollectionLabels is a vector of InputTags      //
      //    from the configuration file.                        //
      ////////////////////////////////////////////////////////////
      // Et & eta distributions
      std::vector<MonitorElement*> etahist;
      std::vector<MonitorElement*> phihist;
      std::vector<MonitorElement*> ethist;
      std::vector<MonitorElement*> etahistmatch;
      std::vector<MonitorElement*> phihistmatch;
      std::vector<MonitorElement*> ethistmatch;
      std::vector<MonitorElement*> histEtOfHltObjMatchToGen;
      std::vector<MonitorElement*> histEtaOfHltObjMatchToGen;
      std::vector<MonitorElement*> histPhiOfHltObjMatchToGen;
      // Plots of efficiency per step
      MonitorElement* total;
      MonitorElement* totalmatch;
      //generator histograms
      MonitorElement* etgen;
      MonitorElement* etagen;
      MonitorElement* phigen;
   
      std::string histName="total_eff";
      std::string histTitle = "total events passing";
      if (!mcMatchedOnly_) {
         // This plot will have bins equal to 2+(number of
         //        HLTCollectionLabels in the config file)
         total = iBooker.book1D(histName.c_str(),histTitle.c_str(),numOfHLTCollectionLabels+2,0,numOfHLTCollectionLabels+2);
         total->setBinLabel(numOfHLTCollectionLabels+1,"Total");
         total->setBinLabel(numOfHLTCollectionLabels+2,"Gen");
         for (unsigned int u=0; u<numOfHLTCollectionLabels; u++) {
            total->setBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());
         }
      }
    
      histName="total_eff_MC_matched";
      histTitle="total events passing (mc matched)";
      totalmatch = iBooker.book1D(histName.c_str(),histTitle.c_str(),numOfHLTCollectionLabels+2,0,numOfHLTCollectionLabels+2);
      totalmatch->setBinLabel(numOfHLTCollectionLabels+1,"Total");
      totalmatch->setBinLabel(numOfHLTCollectionLabels+2,"Gen");
      for (unsigned int u=0; u<numOfHLTCollectionLabels; u++) {
         totalmatch->setBinLabel(u+1,theHLTCollectionLabels[u].label().c_str());
      }
    
      MonitorElement* tmphisto;
      //MonitorElement* tmpiso;
    
      ////////////////////////////////////////////////////////////
      // Set up generator-level histograms                      //
      ////////////////////////////////////////////////////////////
      std::string pdgIdString;
      switch(pdgGen) {
      case 11:
        pdgIdString="Electron";break;
      case 22:
        pdgIdString="Photon";break;
      default:
        pdgIdString="Particle";
      }
    
      histName = "gen_et";
      histTitle= "E_{T} of " + pdgIdString + "s" ;
      etgen =  iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,plotPtMin,plotPtMax);
      histName = "gen_eta";
      histTitle= "#eta of "+ pdgIdString +"s " ;
      etagen = iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,-etaMax_, etaMax_);
      histName = "gen_phi";
      histTitle= "#phi of "+ pdgIdString +"s " ;
      if (!noPhiPlots_) phigen = iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,-phiMax_, phiMax_);
    
      ////////////////////////////////////////////////////////////
      //  Set up histograms of HLT objects                      //
      ////////////////////////////////////////////////////////////
      // Determine what strings to use for histogram titles
      std::vector<std::string> HltHistTitle;
      if ( theHLTCollectionHumanNames.size() == numOfHLTCollectionLabels && useHumanReadableHistTitles_ ) {
        HltHistTitle = theHLTCollectionHumanNames;
      } else {
        for (unsigned int i =0; i < numOfHLTCollectionLabels; i++) {
          HltHistTitle.push_back(theHLTCollectionLabels[i].label());
        }
      }
     
      for(unsigned int i = 0; i< numOfHLTCollectionLabels ; i++){
        if (!mcMatchedOnly_) {
           // Et distribution of HLT objects passing filter i
           histName = theHLTCollectionLabels[i].label()+"et_all";
           histTitle = HltHistTitle[i]+" Et (ALL)";
           tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,plotPtMin,plotPtMax);
           ethist.push_back(tmphisto);
           
           // Eta distribution of HLT objects passing filter i
           histName = theHLTCollectionLabels[i].label()+"eta_all";
           histTitle = HltHistTitle[i]+" #eta (ALL)";
           tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,-etaMax_, etaMax_);
           etahist.push_back(tmphisto);          

           if (!noPhiPlots_) {
             // Phi distribution of HLT objects passing filter i
             histName = theHLTCollectionLabels[i].label()+"phi_all";
             histTitle = HltHistTitle[i]+" #phi (ALL)";
             tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,-phiMax_, phiMax_);
             phihist.push_back(tmphisto);
           }
    
     
           // Et distribution of HLT object that is closest delta-R match to sorted gen particle(s)
           histName  = theHLTCollectionLabels[i].label()+"et";
           histTitle = HltHistTitle[i]+" Et";
           tmphisto  = iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,plotPtMin,plotPtMax);
           histEtOfHltObjMatchToGen.push_back(tmphisto);
    
           // eta distribution of HLT object that is closest delta-R match to sorted gen particle(s)
           histName  = theHLTCollectionLabels[i].label()+"eta";
           histTitle = HltHistTitle[i]+" eta";
           tmphisto  = iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,-etaMax_, etaMax_);
           histEtaOfHltObjMatchToGen.push_back(tmphisto);
    
           if (!noPhiPlots_) {
             // phi distribution of HLT object that is closest delta-R match to sorted gen particle(s)
             histName  = theHLTCollectionLabels[i].label()+"phi";
             histTitle = HltHistTitle[i]+" phi";
             tmphisto  = iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,-phiMax_, phiMax_);
             histPhiOfHltObjMatchToGen.push_back(tmphisto);
           }
       }
    
        // Et distribution of gen object matching HLT object passing filter i
        histName = theHLTCollectionLabels[i].label()+"et_MC_matched";
        histTitle = HltHistTitle[i]+" Et (MC matched)";
        tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,plotPtMin,plotPtMax);
        ethistmatch.push_back(tmphisto);
        
        // Eta distribution of gen object matching HLT object passing filter i
        histName = theHLTCollectionLabels[i].label()+"eta_MC_matched";
        histTitle = HltHistTitle[i]+" #eta (MC matched)";
        tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,-etaMax_, etaMax_);
        etahistmatch.push_back(tmphisto);
    
        if (!noPhiPlots_) {
          // Phi distribution of gen object matching HLT object passing filter i
          histName = theHLTCollectionLabels[i].label()+"phi_MC_matched";
          histTitle = HltHistTitle[i]+" #phi (MC matched)";
          tmphisto =  iBooker.book1D(histName.c_str(),histTitle.c_str(),nbins_,-phiMax_, phiMax_);
          phihistmatch.push_back(tmphisto);
        }
      }

      // Et & eta distributions
      etahists.push_back(etahist);
      phihists.push_back(phihist);
      ethists.push_back(ethist);
      etahistmatchs.push_back(etahistmatch);
      phihistmatchs.push_back(phihistmatch);
      ethistmatchs.push_back(ethistmatch);
      histEtOfHltObjMatchToGens.push_back(histEtOfHltObjMatchToGen);
      histEtaOfHltObjMatchToGens.push_back(histEtaOfHltObjMatchToGen);
      histPhiOfHltObjMatchToGens.push_back(histPhiOfHltObjMatchToGen);
      // commented out because uses data not included in HTLDEBUG and uses
      // Isolation distributions
      //etahistisos.push_back(etahistiso);
      //phihistisos.push_back(phihistiso);
      //ethistisos.push_back(ethistiso);
      //etahistisomatchs.push_back(etahistisomatch);
      //phihistisomatchs.push_back(phihistisomatch);
      //ethistisomatchs.push_back(ethistisomatch);
      //histEtIsoOfHltObjMatchToGens.push_back(histEtIsoOfHltObjMatchToGen); 
      //histEtaIsoOfHltObjMatchToGens.push_back(histEtaIsoOfHltObjMatchToGen);
      //histPhiIsoOfHltObjMatchToGens.push_back(histPhiIsoOfHltObjMatchToGen);

      totals.push_back(total);
      totalmatchs.push_back(totalmatch);
      etgens.push_back(etgen);
      etagens.push_back(etagen);
      phigens.push_back(phigen);
   }
}

////////////////////////////////////////////////////////////////////////////////
//                                Destructor                                  //
////////////////////////////////////////////////////////////////////////////////
EmDQM::~EmDQM(){
}
////////////////////////////////////////////////////////////////////////////////

bool EmDQM::checkGeneratedParticlesRequirement(const edm::Event & event)
{
  ////////////////////////////////////////////////////////////
   // Decide if this was an event of interest.               //
   //  Did the highest energy particles happen               //
   //  to have |eta| < 2.5 ?  Then continue.                 //
   ////////////////////////////////////////////////////////////
   edm::Handle< edm::View<reco::Candidate> > genParticles;
   event.getByToken(genParticles_token, genParticles);
   if(!genParticles.isValid()) {
     if (verbosity_ >= OUTPUT_WARNINGS)
        edm::LogWarning("EmDQM") << "genParticles invalid.";
     return false;
   }

   std::vector<reco::LeafCandidate> allSortedGenParticles;

   for(edm::View<reco::Candidate>::const_iterator currentGenParticle = genParticles->begin(); currentGenParticle != genParticles->end(); currentGenParticle++){

     // TODO: do we need to check the states here again ?
     // in principle, there should collections produced with the python configuration
     // (other than 'genParticles') which fulfill these criteria
     if (  !( abs((*currentGenParticle).pdgId())==pdgGen  && (*currentGenParticle).status()==1 && (*currentGenParticle).et() > 2.0)  )  continue;

     reco::LeafCandidate tmpcand( *(currentGenParticle) );

     if (tmpcand.et() < plotEtMin) continue;

     allSortedGenParticles.push_back(tmpcand);
   }

   std::sort(allSortedGenParticles.begin(), allSortedGenParticles.end(),pTGenComparator_);

   // return false if not enough particles found
   if (allSortedGenParticles.size() < gencut_)
     return false;

   // additional check (this might be legacy code and we need to check
   // whether this should not be removed ?)

   // We now have a sorted collection of all generated particles
   // with pdgId = pdgGen.
   // Loop over them to see if the top gen particles have eta within acceptance
  // bool keepEvent = true;
   for (unsigned int i = 0 ; i < gencut_ ; i++ ) {
     bool inECALgap = fabs(allSortedGenParticles[i].eta()) > 1.4442 && fabs(allSortedGenParticles[i].eta()) < 1.556;
     if ( (fabs(allSortedGenParticles[i].eta()) > genEtaAcc_) || inECALgap ) {
       //edm::LogWarning("EmDQM") << "Throwing event away. Gen particle with pdgId="<< allSortedGenParticles[i].pdgId() <<"; et="<< allSortedGenParticles[i].et() <<"; and eta="<< allSortedGenParticles[i].eta() <<" beyond acceptance.";
       return false;
     }
   }

   // all tests passed
   return true;
}
////////////////////////////////////////////////////////////////////////////////

bool EmDQM::checkRecoParticlesRequirement(const edm::Event & event)
{
  // note that this code is very similar to the one in checkGeneratedParticlesRequirement(..)
  // and hopefully can be merged with it at some point in the future

  edm::Handle< edm::View<reco::Candidate> > referenceParticles;
  // get the right data according to the trigger path currently looked at
  if (autoConfMode_) {
    switch(reqNum) {
       case 1:
          if (pdgGen == 11) event.getByToken(gencutColl_fidWenu_token, referenceParticles);
          else event.getByToken(gencutColl_fidGammaJet_token, referenceParticles);
          break;
       case 2:
          if (pdgGen == 11) event.getByToken(gencutColl_fidZee_token, referenceParticles);
          else event.getByToken(gencutColl_fidDiGamma_token, referenceParticles);
          break;
       case 3:
          event.getByToken(gencutColl_fidTripleEle_token, referenceParticles);
          break;
    }
  } else {
    event.getByToken(gencutColl_manualConf_token, referenceParticles);
  }
  if(!referenceParticles.isValid()) {
     if (verbosity_ >= OUTPUT_WARNINGS)
        edm::LogWarning("EmDQM") << "referenceParticles invalid.";
     return false;
  }

  std::vector<const reco::Candidate *> allSortedReferenceParticles;

  for(edm::View<reco::Candidate>::const_iterator currentReferenceParticle = referenceParticles->begin();
      currentReferenceParticle != referenceParticles->end();
      currentReferenceParticle++)
  {
     if ( currentReferenceParticle->et() <= 2.0)
       continue;

     // Note that for determining the overall efficiency,
     // we should only allow
     //
     // HOWEVER: for turn-on curves, we need to let
     //          more electrons pass
     if (currentReferenceParticle->et() < plotEtMin)
       continue;

     // TODO: instead of filling a new vector we could simply count here...
     allSortedReferenceParticles.push_back(&(*currentReferenceParticle));
  }

   // std::sort(allSortedReferenceParticles.begin(), allSortedReferenceParticles.end(),pTComparator_);

   // return false if not enough particles found
   return allSortedReferenceParticles.size() >= gencut_;
}

////////////////////////////////////////////////////////////////////////////////
//                     method called to for each event                        //
////////////////////////////////////////////////////////////////////////////////
void 
EmDQM::analyze(const edm::Event & event , const edm::EventSetup& setup)
{
  // loop over all the trigger path parameter sets
  unsigned int vPos = 0;
  for (std::vector<edm::ParameterSet>::iterator psetIt = paramSets.begin(); psetIt != paramSets.end(); ++psetIt, ++vPos) {
    SetVarsFromPSet(psetIt);
    // get the set forthe current path
    hltCollectionLabelsFound = hltCollectionLabelsFoundPerPath.at(vPos);
    hltCollectionLabelsMissed = hltCollectionLabelsMissedPerPath.at(vPos);

    ////////////////////////////////////////////////////////////
    //           Check if there's enough gen particles        //
    //             of interest                                //
    ////////////////////////////////////////////////////////////
    // get the right data according to the trigger path currently looked at
    edm::Handle< edm::View<reco::Candidate> > cutCounter;
    if (autoConfMode_) {
      switch(reqNum) {
         case 1:
            if (pdgGen == 11) event.getByToken(gencutColl_fidWenu_token, cutCounter);
            else event.getByToken(gencutColl_fidGammaJet_token, cutCounter);
            break;
         case 2:
            if (pdgGen == 11) event.getByToken(gencutColl_fidZee_token, cutCounter);
            else event.getByToken(gencutColl_fidDiGamma_token, cutCounter);
            break;
         case 3:
            event.getByToken(gencutColl_fidTripleEle_token, cutCounter);
            break;
      }
    } else {
      event.getByToken(gencutColl_manualConf_token, cutCounter);
    }
    if (cutCounter->size() < (unsigned int)gencut_) {
      //edm::LogWarning("EmDQM") << "Less than "<< gencut_ <<" gen particles with pdgId=" << pdgGen;
      continue;
    }

    // fill L1 and HLT info
    // get objects possed by each filter
    edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
    event.getByToken(triggerObject_token,triggerObj);
    if(!triggerObj.isValid()) {
      if (verbosity_ >= OUTPUT_WARNINGS)
         edm::LogWarning("EmDQM") << "parameter triggerobject (" << triggerObject_ << ") does not corresond to a valid TriggerEventWithRefs product. Please check especially the process name (e.g. when running over reprocessed datasets)";
      continue;
    }

    // Were enough high energy gen particles found?
    if (event.isRealData()) {
      // running validation on data.
      // TODO: we should check that the entire
      //       run is on the same type (all data or
      //       all MC). Otherwise one gets
      //       uninterpretable results...
      if (!checkRecoParticlesRequirement(event))
        continue;
    }
    else { // MC
      // if no, throw event away
      if (!checkGeneratedParticlesRequirement(event))
        continue;
    }

    // It was an event worth keeping. Continue.

    ////////////////////////////////////////////////////////////
    //  Fill the bin labeled "Total"                          //
    //   This will be the number of events looked at.         //
    ////////////////////////////////////////////////////////////
    if (!mcMatchedOnly_) totals.at(vPos)->Fill(numOfHLTCollectionLabels+0.5);
    totalmatchs.at(vPos)->Fill(numOfHLTCollectionLabels+0.5);

    ////////////////////////////////////////////////////////////
    //               Fill generator info                      //
    ////////////////////////////////////////////////////////////
    // the gencut_ highest Et generator objects of the preselected type are our matches

    std::vector<reco::Particle> sortedGen;
    for(edm::View<reco::Candidate>::const_iterator genpart = cutCounter->begin(); genpart != cutCounter->end();genpart++){
      reco::Particle tmpcand(  genpart->charge(), genpart->p4(), genpart->vertex(),genpart->pdgId(),genpart->status() );
      if (tmpcand.et() >= plotEtMin) {
        sortedGen.push_back(tmpcand);
      }
    }
    std::sort(sortedGen.begin(),sortedGen.end(),pTComparator_ );

    // Now the collection of gen particles is sorted by pt.
    // So, remove all particles from the collection so that we 
    // only have the top "1 thru gencut_" particles in it
    if (sortedGen.size() < gencut_){
      continue;
    }
    sortedGen.erase(sortedGen.begin()+gencut_,sortedGen.end());

    for (unsigned int i = 0 ; i < gencut_ ; i++ ) {
      etgens.at(vPos)->Fill( sortedGen[i].et()  ); //validity has been implicitily checked by the cut on gencut_ above
      if (sortedGen[i].et() > minEtForEtaEffPlot_) {
        etagens.at(vPos)->Fill( sortedGen[i].eta() );
        if (!noPhiPlots_) phigens.at(vPos)->Fill( sortedGen[i].phi() );
      }
    } // END of loop over Generated particles
    if (gencut_ >= reqNum && !mcMatchedOnly_) totals.at(vPos)->Fill(numOfHLTCollectionLabels+1.5); // this isn't really needed anymore keep for backward comp.
    if (gencut_ >= reqNum) totalmatchs.at(vPos)->Fill(numOfHLTCollectionLabels+1.5); // this isn't really needed anymore keep for backward comp.
            
    bool accepted = true;  // flags that the event has been accepted by all filters before
    edm::Handle<edm::TriggerResults> hltResults;
    event.getByToken(hltResults_token, hltResults);
    ////////////////////////////////////////////////////////////
    //            Loop over filter modules                    //
    ////////////////////////////////////////////////////////////
    for(unsigned int n=0; n < numOfHLTCollectionLabels ; n++) {
      // check that there are not less sortedGen particles than nCandCut requires for this filter
      if (sortedGen.size() < nCandCuts.at(n)) {
         if (verbosity_ >= OUTPUT_ERRORS)
            edm::LogError("EmDQM") << "There are less generated particles than the module '" << theHLTCollectionLabels[n].label() << "' requires.";
         continue;
      }
      std::vector<reco::Particle> sortedGenForFilter(sortedGen);
      sortedGenForFilter.erase(sortedGenForFilter.begin() + nCandCuts.at(n), sortedGenForFilter.end());

      // Fill only if this filter was run.
      if (pathIndex != 0 && hltConfig_.moduleIndex(pathIndex, theHLTCollectionLabels[n].label()) > hltResults->index(pathIndex)) break;
      // These numbers are from the Parameter Set, such as:
      //   theHLTOutputTypes = cms.uint32(100)
      switch(theHLTOutputTypes[n]) 
      {
        case trigger::TriggerL1NoIsoEG: // Non-isolated Level 1
          histoFillerL1NonIso->fillHistos(triggerObj,event,vPos,n,sortedGenForFilter,accepted);
          break;
        case trigger::TriggerL1IsoEG: // Isolated Level 1
          histoFillerL1Iso->fillHistos(triggerObj,event,vPos,n,sortedGenForFilter,accepted);
          break;
        case trigger::TriggerPhoton: // Photon 
          histoFillerPho->fillHistos(triggerObj,event,vPos,n,sortedGenForFilter,accepted);
          break;
        case trigger::TriggerElectron: // Electron 
          histoFillerEle->fillHistos(triggerObj,event,vPos,n,sortedGenForFilter,accepted);
          break;
        case trigger::TriggerCluster: // TriggerCluster
          histoFillerClu->fillHistos(triggerObj,event,vPos,n,sortedGenForFilter,accepted);
          break;
        default: 
          throw(cms::Exception("Release Validation Error") << "HLT output type not implemented: theHLTOutputTypes[n]" );
      }
    } // END of loop over filter modules

    // earse the dummy and fill with real set
    hltCollectionLabelsFoundPerPath.at(vPos) = hltCollectionLabelsFound;
    hltCollectionLabelsMissedPerPath.at(vPos) = hltCollectionLabelsMissed;
  }
}

////////////////////////////////////////////////////////////////////////////////
// fillHistos                                                                 //
//   Called by analyze method.                                                //
////////////////////////////////////////////////////////////////////////////////
template <class T> void HistoFiller<T>::fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& triggerObj,const edm::Event& iEvent ,unsigned int vPos, unsigned int n,std::vector<reco::Particle>& sortedGen, bool &accepted)
{
  std::vector<edm::Ref<T> > recoecalcands;
  if ( ( triggerObj->filterIndex(dqm->theHLTCollectionLabels[n])>=triggerObj->size() )){ // only process if available
    dqm->hltCollectionLabelsMissed.insert(dqm->theHLTCollectionLabels[n].encode());
    accepted = false;
    return;
  }

  dqm->hltCollectionLabelsFound.insert(dqm->theHLTCollectionLabels[n].encode());

  ////////////////////////////////////////////////////////////
  //      Retrieve saved filter objects                     //
  ////////////////////////////////////////////////////////////
  triggerObj->getObjects(triggerObj->filterIndex(dqm->theHLTCollectionLabels[n]),dqm->theHLTOutputTypes[n],recoecalcands);
  //Danger: special case, L1 non-isolated
  // needs to be merged with L1 iso
  if (dqm->theHLTOutputTypes[n] == trigger::TriggerL1NoIsoEG){
    std::vector<edm::Ref<T> > isocands;
    triggerObj->getObjects(triggerObj->filterIndex(dqm->theHLTCollectionLabels[n]),trigger::TriggerL1IsoEG,isocands);
    if (isocands.size()>0) 
      {
        for (unsigned int i=0; i < isocands.size(); i++)
          recoecalcands.push_back(isocands[i]);
      }
  } // END of if theHLTOutputTypes == 82
  

  if (recoecalcands.size() < 1){ // stop if no object passed the previous filter
    accepted = false;
    return;
  }

  //if (recoecalcands.size() >= reqNum ) 
  if (recoecalcands.size() >= dqm->nCandCuts.at(n) && !dqm->mcMatchedOnly_) 
    dqm->totals.at(vPos)->Fill(n+0.5);

  ///////////////////////////////////////////////////
  // check for validity                            //
  // prevents crash in CMSSW_3_1_0_pre6            //
  ///////////////////////////////////////////////////
  for (unsigned int j=0; j<recoecalcands.size(); j++){
    if(!( recoecalcands.at(j).isAvailable())){
      if (dqm->verbosity_ >= dqm->OUTPUT_ERRORS)
         edm::LogError("EmDQMInvalidRefs") << "Event content inconsistent: TriggerEventWithRefs contains invalid Refs. Invalid refs for: " << dqm->theHLTCollectionLabels[n].label() << ". The collection that this module uses may has been dropped in the event.";
      return;
    }
  }

  if (!dqm->mcMatchedOnly_) {
    ////////////////////////////////////////////////////////////
    // Loop over the Generated Particles, and find the        //
    // closest HLT object match.                              //
    ////////////////////////////////////////////////////////////
    //for (unsigned int i=0; i < gencut_; i++) {
    for (unsigned int i=0; i < dqm->nCandCuts.at(n); i++) {
      math::XYZVector currentGenParticleMomentum = sortedGen[i].momentum();

      float closestDeltaR = 0.5;
      int closestEcalCandIndex = -1;
      for (unsigned int j=0; j<recoecalcands.size(); j++) {
        float deltaR = DeltaR(recoecalcands[j]->momentum(),currentGenParticleMomentum);

        if (deltaR < closestDeltaR) {
          closestDeltaR = deltaR;
          closestEcalCandIndex = j;
        }
      }

      // If an HLT object was found within some delta-R
      // of this gen particle, store it in a histogram
      if ( closestEcalCandIndex >= 0 ) {
        dqm->histEtOfHltObjMatchToGens.at(vPos).at(n)->Fill( recoecalcands[closestEcalCandIndex]->et()  );
        dqm->histEtaOfHltObjMatchToGens.at(vPos).at(n)->Fill( recoecalcands[closestEcalCandIndex]->eta() );
        if (!dqm->noPhiPlots_) dqm->histPhiOfHltObjMatchToGens.at(vPos).at(n)->Fill( recoecalcands[closestEcalCandIndex]->phi() );
        
      } // END of if closestEcalCandIndex >= 0
    }

    ////////////////////////////////////////////////////////////
    //  Loop over all HLT objects in this filter step, and    //
    //  fill histograms.                                      //
    ////////////////////////////////////////////////////////////
    //  bool foundAllMatches = false;
    //  unsigned int numOfHLTobjectsMatched = 0;
    for (unsigned int i=0; i<recoecalcands.size(); i++) {
      //// See if this HLT object has a gen-level match
      //float closestGenParticleDr = 99.0;
      //for(unsigned int j =0; j < gencut_; j++) {
      //  math::XYZVector currentGenParticle = sortedGen[j].momentum();

      //  double currentDeltaR = DeltaR(recoecalcands[i]->momentum(),currentGenParticle);
      //  if ( currentDeltaR < closestGenParticleDr ) {
      //    closestGenParticleDr = currentDeltaR;
      //  }
      //}
      //// If this HLT object did not have a gen particle match, go to next HLT object
      //if ( !(fabs(closestGenParticleDr < 0.3)) ) continue;
   
      //numOfHLTobjectsMatched++;
      //if (numOfHLTobjectsMatched >= gencut_) foundAllMatches=true;

      // Fill HLT object histograms
      dqm->ethists.at(vPos).at(n) ->Fill(recoecalcands[i]->et() );
      dqm->etahists.at(vPos).at(n)->Fill(recoecalcands[i]->eta() );
      if (!dqm->noPhiPlots_) dqm->phihists.at(vPos).at(n)->Fill(recoecalcands[i]->phi() );

    }
  }

  ////////////////////////////////////////////////////////////
  //        Fill mc matched objects into histograms         //
  ////////////////////////////////////////////////////////////
  unsigned int matchedMcParts = 0;
  float mindist=0.3;
  if(n==0) mindist=0.5; //low L1-resolution => allow wider matching 
  for(unsigned int i =0; i < dqm->nCandCuts.at(n); ++i){
    //match generator candidate
    bool matchThis= false;
    math::XYZVector candDir=sortedGen[i].momentum();
    //unsigned int closest = 0;
    double closestDr = 1000.;
    for(unsigned int trigOb = 0 ; trigOb < recoecalcands.size(); ++trigOb){
      double dr = DeltaR(recoecalcands[trigOb]->momentum(),candDir);
      if (dr < closestDr) {
        closestDr = dr;
        //closest = trigOb;
      }
      if (closestDr > mindist) { // it's not really a "match" if it's that far away
        //closest = -1;
      } else {
        matchedMcParts++;
        matchThis = true;
      }
    }
    if ( !matchThis ) {
      accepted = false;
      continue; // only plot matched candidates
    }
    // fill coordinates of mc particle matching trigger object
    dqm->ethistmatchs.at(vPos).at(n) ->Fill( sortedGen[i].et()  );
    if (sortedGen[i].et() > dqm->minEtForEtaEffPlot_) {
      dqm->etahistmatchs.at(vPos).at(n)->Fill( sortedGen[i].eta() );
      if (!dqm->noPhiPlots_) dqm->phihistmatchs.at(vPos).at(n)->Fill( sortedGen[i].phi() );
    }
  }
  // fill total mc matched efficiency

  if (matchedMcParts >= dqm->nCandCuts.at(n) && accepted == true)
    dqm->totalmatchs.at(vPos)->Fill(n+0.5);
}

void 
EmDQM::endRun(edm::Run const &iRun, edm::EventSetup const &iSetup)
{
   // loop over all the trigger path parameter sets
   unsigned int vPos = 0;
   for (std::vector<edm::ParameterSet>::iterator psetIt = paramSets.begin(); psetIt != paramSets.end(); ++psetIt, ++vPos) {
      SetVarsFromPSet(psetIt);

      // print information about hltCollectionLabels which were not found
      // (but only those which were never found)

      // check which ones were never found
      std::vector<std::string> labelsNeverFound;
      
      BOOST_FOREACH(const edm::InputTag &tag, hltCollectionLabelsMissedPerPath.at(vPos))
      {
        if ((hltCollectionLabelsFoundPerPath.at(vPos)).count(tag.encode()) == 0)
          // never found
          labelsNeverFound.push_back(tag.encode());

      } // loop over all tags which were missed at least once

      if (labelsNeverFound.empty())
        continue;

      std::sort(labelsNeverFound.begin(), labelsNeverFound.end());

      // there was at least one label which was never found
      // (note that this could also be because the corresponding
      // trigger path slowly fades out to zero efficiency)
      if (verbosity_ >= OUTPUT_WARNINGS)
         edm::LogWarning("EmDQM") << "There were some HLTCollectionLabels which were never found:";

      BOOST_FOREACH(const edm::InputTag &tag, labelsNeverFound)
      {
        if (verbosity_ >= OUTPUT_ALL)
           edm::LogPrint("EmDQM") << "  " << tag;
      }
   }
}

//////////////////////////////////////////////////////////////////////////////// 
//      method called once each job just after ending the event loop          //
//////////////////////////////////////////////////////////////////////////////// 
void EmDQM::endJob()
{

}

// returns count of non-overlapping occurrences of 'sub' in 'str'
int 
EmDQM::countSubstring(const std::string& str, const std::string& sub)
{
    if (sub.length() == 0) return 0;
    int count = 0;
    for (size_t offset = str.find(sub); offset != std::string::npos;
	 offset = str.find(sub, offset + sub.length()))
    {
        ++count;
    }
    return count;
}

//----------------------------------------------------------------------
// find egamma trigger paths from trigger name
std::vector<std::vector<std::string> >
EmDQM::findEgammaPaths()
{
   std::vector<std::vector<std::string> > Paths(5);
   // Loop over all paths in the menu
   for (unsigned int i=0; i<hltConfig_.size(); i++) {

      std::string path = hltConfig_.triggerName(i);

      // Find electron and photon paths and classify them
      if (int(path.find("HLT_")) == 0) {    // Path should start with 'HLT_'

         int scCount = countSubstring(path, "_SC");
         int eleCount = countSubstring(path, "Ele");
         int doubleEleCount = countSubstring(path, "DoubleEle");
         int tripleEleCount = countSubstring(path, "TripleEle");
         int photonCount = countSubstring(path, "Photon");
         int doublePhotonCount = countSubstring(path, "DoublePhoton");

         int totEleCount = 2*tripleEleCount + doubleEleCount + eleCount + scCount;
         int totPhotonCount = doublePhotonCount + photonCount;

         if (totEleCount + totPhotonCount < 1) continue;
         switch (totEleCount) {
            case 1:
               Paths[TYPE_SINGLE_ELE].push_back(path);
               //std::cout << "Electron \t" << path << std::endl;
               break;
            case 2:
               Paths[TYPE_DOUBLE_ELE].push_back(path);
               //std::cout << "DoubleElectron \t" << path << std::endl;
               break;
            case 3:
               Paths[TYPE_TRIPLE_ELE].push_back(path);
               //std::cout << "TripleElectron \t" << path << std::endl;
               break;
         }

         switch (totPhotonCount) {
            case 1:
               Paths[TYPE_SINGLE_PHOTON].push_back(path);
               //std::cout << "Photon \t\t" << path << std::endl;
               break;
            case 2:
               Paths[TYPE_DOUBLE_PHOTON].push_back(path);
               //std::cout << "DoublePhoton \t" << path << std::endl;
               break;
         }
      }
      //std::cout << i << " triggerName: " << path << " containing " << hltConfig_.size(i) << " modules."<< std::endl;
   }
   return Paths;
}

//----------------------------------------------------------------------
// get the names of filters of a given path
std::vector<std::string>
EmDQM::getFilterModules(const std::string& path)
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
// get the primary Et cut from the trigger name
double
EmDQM::getPrimaryEtCut(const std::string& path)
{
   double minEt = -1;

   boost::regex reg("^HLT_.*?(Ele|Photon|EG|SC)([[:digit:]]+).*");

   boost::smatch what;
   if (boost::regex_match(path, what, reg, boost::match_extra))
   {
     minEt = boost::lexical_cast<double>(what[2]); 
   }

   return minEt;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQM::makePSetForL1SeedFilter(const std::string& moduleName)
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

  int orCount = countSubstring(moduleName, "OR");
  int egCount = countSubstring(moduleName, "EG");
  int dEgCount = countSubstring(moduleName, "DoubleEG");
  int tEgCount = countSubstring(moduleName, "TripleEG");

  int candCount = 2*tEgCount + dEgCount + egCount;
  // if L1 is and OR of triggers try the assumption that all of them are similar first and, if not successful, let the path name decide
  if (orCount > 0 && candCount > 0) {
     if (egCount % (orCount+1) == 0 && dEgCount % (orCount+1) == 0 && tEgCount % (orCount+1) == 0) candCount /= (orCount+1);
     else candCount = -1;
  }

  switch (candCount) {
     case 0:
        retPSet.addParameter<int>("ncandcut", 0);
        break;
     case 1:
        retPSet.addParameter<int>("ncandcut", 1);
        break;
     case 2:
        retPSet.addParameter<int>("ncandcut", 2);
        break;
     case 3:
        retPSet.addParameter<int>("ncandcut", 3);
        break;
     default:
        retPSet.addParameter<int>("ncandcut", -1);
  }

  return retPSet;
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQM::makePSetForL1SeedToSuperClusterMatchFilter(const std::string& moduleName)
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
EmDQM::makePSetForEtFilter(const std::string& moduleName)
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
EmDQM::makePSetForOneOEMinusOneOPFilter(const std::string& moduleName)
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
EmDQM::makePSetForPixelMatchFilter(const std::string& moduleName)
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
EmDQM::makePSetForEgammaDoubleEtDeltaPhiFilter(const std::string& moduleName)
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
EmDQM::makePSetForEgammaGenericFilter(const std::string& moduleName)
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
  //  edm::LogError("EmDQM") << "nonIsoTag of HLTEgammaGenericFilter '" << moduleName <<  "' is empty.";
  //  return retPSet;
  //}
  //if (inputType != hltConfig_.moduleType(nonIsoTag.label())) {
  //  edm::LogError("EmDQM") << "C++ Type of isoTag '" << inputType << "' and nonIsoTag '" << hltConfig_.moduleType(nonIsoTag.label()) << "' are not the same for HLTEgammaGenericFilter '" << moduleName <<  "'.";
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
     edm::LogError("EmDQM") << "Can't determine what the HLTEgammaGenericFilter '" << moduleName <<  "' should do: uses a collection produced by a module of C++ type '" << inputType << "'.";
  return edm::ParameterSet();
}

//----------------------------------------------------------------------

edm::ParameterSet 
EmDQM::makePSetForEgammaGenericQuadraticFilter(const std::string& moduleName)
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
  //  edm::LogError("EmDQM") << "nonIsoTag of HLTEgammaGenericFilter '" << moduleName <<  "' is empty.";
  //  return retPSet;
  //}
  //if (inputType != hltConfig_.moduleType(nonIsoTag.label())) {
  //  edm::LogError("EmDQM") << "C++ Type of isoTag '" << inputType << "' and nonIsoTag '" << hltConfig_.moduleType(nonIsoTag.label()) << "' are not the same for HLTEgammaGenericFilter '" << moduleName <<  "'.";
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
     edm::LogError("EmDQM") << "Can't determine what the HLTEgammaGenericQuadraticFilter '" << moduleName <<  "' should do: uses a collection produced by a module of C++ type '" << inputType << "'.";
  return edm::ParameterSet();
}

//----------------------------------------------------------------------

edm::ParameterSet
EmDQM::makePSetForElectronGenericFilter(const std::string& moduleName)
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
  //  edm::LogError("EmDQM") << "nonIsoTag of HLTElectronGenericFilter '" << moduleName <<  "' is empty.";
  //  return retPSet;
  //}
  //if (inputType != hltConfig_.moduleType(nonIsoTag.label())) {
  //  edm::LogError("EmDQM") << "C++ Type of isoTag '" << inputType << "' and nonIsoTag '" << hltConfig_.moduleType(nonIsoTag.label()) << "' are not the same for HLTElectronGenericFilter '" << moduleName <<  "'.";
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
     edm::LogError("EmDQM") << "Can't determine what the HLTElectronGenericFilter '" << moduleName <<  "' should do: uses a collection produced by a module of C++ type '" << inputType << "'.";
  return edm::ParameterSet();
}

//----------------------------------------------------------------------

void EmDQM::SetVarsFromPSet(std::vector<edm::ParameterSet>::iterator psetIt) 
{
  dirname_="HLT/HLTEgammaValidation/"+psetIt->getParameter<std::string>("@module_label");
 
  pathIndex = psetIt->getUntrackedParameter<unsigned int>("pathIndex", 0);
  // parameters for generator study
  reqNum    = psetIt->getParameter<unsigned int>("reqNum");
  pdgGen    = psetIt->getParameter<int>("pdgGen");
  // plotting parameters (untracked because they don't affect the physics)
  plotEtMin  = psetIt->getUntrackedParameter<double>("genEtMin",0.);
  plotPtMin  = psetIt->getUntrackedParameter<double>("PtMin",0.);
  plotPtMax  = psetIt->getUntrackedParameter<double>("PtMax",1000.);
 
  //preselction cuts 
  gencutCollection_= psetIt->getParameter<edm::InputTag>("cutcollection");
  gencut_          = psetIt->getParameter<int>("cutnum");
 
  ////////////////////////////////////////////////////////////
  //         Read in the Vector of Parameter Sets.          //
  //           Information for each filter-step             //
  ////////////////////////////////////////////////////////////
  std::vector<edm::ParameterSet> filters = 
       psetIt->getParameter<std::vector<edm::ParameterSet> >("filters");
 
  // empty vectors of parameters from previous paths
  theHLTCollectionLabels.clear();
  theHLTOutputTypes.clear();
  theHLTCollectionHumanNames.clear();
  plotBounds.clear();
  isoNames.clear();
  plotiso.clear();
  nCandCuts.clear();

  int i = 0;
  for(std::vector<edm::ParameterSet>::iterator filterconf = filters.begin() ; filterconf != filters.end() ; filterconf++)
  {
 
    theHLTCollectionLabels.push_back(filterconf->getParameter<edm::InputTag>("HLTCollectionLabels"));
    theHLTOutputTypes.push_back(filterconf->getParameter<int>("theHLTOutputTypes"));
    // Grab the human-readable name, if it is not specified, use the Collection Label
    theHLTCollectionHumanNames.push_back(filterconf->getUntrackedParameter<std::string>("HLTCollectionHumanName",theHLTCollectionLabels[i].label()));
 
    std::vector<double> bounds = filterconf->getParameter<std::vector<double> >("PlotBounds");
    // If the size of plot "bounds" vector != 2, abort
    assert(bounds.size() == 2);
    plotBounds.push_back(std::pair<double,double>(bounds[0],bounds[1]));
    isoNames.push_back(filterconf->getParameter<std::vector<edm::InputTag> >("IsoCollections"));

    //for (unsigned int i=0; i<isoNames.back().size(); i++) {
    //  switch(theHLTOutputTypes.back())  {
    //  case trigger::TriggerL1NoIsoEG:
    //    histoFillerL1NonIso->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<l1extra::L1EmParticleCollection , float>>>(isoNames.back()[i]));
    //    break;
    //  case trigger::TriggerL1IsoEG: // Isolated Level 1
    //    histoFillerL1Iso->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<l1extra::L1EmParticleCollection , float>>>(isoNames.back()[i]));
    //    break;
    //  case trigger::TriggerPhoton: // Photon 
    //    histoFillerPho->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<reco::RecoEcalCandidateCollection , float>>>(isoNames.back()[i]));
    //    break;
    //  case trigger::TriggerElectron: // Electron 
    //    histoFillerEle->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<reco::ElectronCollection , float>>>(isoNames.back()[i]));
    //    break;
    //  case trigger::TriggerCluster: // TriggerCluster
    //    histoFillerClu->isoNameTokens_.push_back(consumes<edm::AssociationMap<edm::OneToValue<reco::RecoEcalCandidateCollection , float>>>(isoNames.back()[i]));
    //    break;
    //  default:
    //    throw(cms::Exception("Release Validation Error") << "HLT output type not implemented: theHLTOutputTypes[n]" );
    //  }
    //}

    // If the size of the isoNames vector is not greater than zero, abort
    assert(isoNames.back().size()>0);
    if (isoNames.back().at(0).label()=="none") {
      plotiso.push_back(false);
    } else {
      if (!noIsolationPlots_) plotiso.push_back(true);
      else plotiso.push_back(false);
    }
    nCandCuts.push_back(filterconf->getParameter<int>("ncandcut"));
    i++;
  } // END of loop over parameter sets
 
  // Record number of HLTCollectionLabels
  numOfHLTCollectionLabels = theHLTCollectionLabels.size();
}

//----------------------------------------------------------------------

DEFINE_FWK_MODULE(EmDQM);
