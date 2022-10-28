
/** \file HLTHiggsSubAnalysis.cc
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"

#include "HLTHiggsSubAnalysis.h"
#include "EVTColContainer.h"
#include "MatchStruct.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "TPRegexp.h"
#include "TString.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "TPRegexp.h"
#include "TRegexp.h"
#include "TString.h"

#include <set>
#include <algorithm>

HLTHiggsSubAnalysis::HLTHiggsSubAnalysis(const edm::ParameterSet& pset,
                                         const std::string& analysisname,
                                         edm::ConsumesCollector&& iC)
    : _pset(pset),
      _analysisname(analysisname),
      _minCandidates(0),
      _HtJetPtMin(0),
      _HtJetEtaMax(0),
      _hltProcessName(pset.getParameter<std::string>("hltProcessName")),
      _histDirectory(pset.getParameter<std::string>("histDirectory")),
      _genParticleLabel(iC.consumes<reco::GenParticleCollection>(pset.getParameter<std::string>("genParticleLabel"))),
      _genJetLabel(iC.consumes<reco::GenJetCollection>(pset.getParameter<std::string>("genJetLabel"))),
      _recoHtJetLabel(iC.consumes<reco::PFJetCollection>(
          pset.getUntrackedParameter<std::string>("recoHtJetLabel", "ak4PFJetsCHS"))),
      _parametersEta(pset.getParameter<std::vector<double>>("parametersEta")),
      _parametersPhi(pset.getParameter<std::vector<double>>("parametersPhi")),
      _parametersPu(pset.getParameter<std::vector<double>>("parametersPu")),
      _parametersHt(0),
      _parametersTurnOn(pset.getParameter<std::vector<double>>("parametersTurnOn")),
      _trigResultsTag(iC.consumes<edm::TriggerResults>(edm::InputTag("TriggerResults", "", _hltProcessName))),
      _genJetSelector(nullptr),
      _recMuonSelector(nullptr),
      _recElecSelector(nullptr),
      _recCaloMETSelector(nullptr),
      _recPFMETSelector(nullptr),
      _recPFTauSelector(nullptr),
      _recPhotonSelector(nullptr),
      _recPFJetSelector(nullptr),
      _recTrackSelector(nullptr),
      _NminOneCuts(0),
      _useNminOneCuts(false) {
  // Specific parameters for this analysis
  edm::ParameterSet anpset = pset.getParameter<edm::ParameterSet>(analysisname);
  // Collections labels (but genparticles already initialized)
  // initializing _recLabels data member)
  if (anpset.exists("parametersTurnOn")) {
    _parametersTurnOn = anpset.getParameter<std::vector<double>>("parametersTurnOn");
    _pset.addParameter("parametersTurnOn", _parametersTurnOn);
  }
  this->bookobjects(anpset, iC);
  // Generic objects: Initialization of cuts
  for (std::map<unsigned int, std::string>::const_iterator it = _recLabels.begin(); it != _recLabels.end(); ++it) {
    const std::string objStr = EVTColContainer::getTypeString(it->first);
    _genCut[it->first] = pset.getParameter<std::string>(objStr + "_genCut");
    _recCut[it->first] = pset.getParameter<std::string>(objStr + "_recCut");
    _cutMinPt[it->first] = pset.getParameter<double>(objStr + "_cutMinPt");
    _cutMaxEta[it->first] = pset.getParameter<double>(objStr + "_cutMaxEta");
  }
  //--- Updating parameters if has to be modified for this particular specific analysis
  for (std::map<unsigned int, std::string>::const_iterator it = _recLabels.begin(); it != _recLabels.end(); ++it) {
    const std::string objStr = EVTColContainer::getTypeString(it->first);
    if (anpset.existsAs<std::string>(objStr + "_genCut", false)) {
      _genCut[it->first] = anpset.getUntrackedParameter<std::string>(objStr + "_genCut");
    }
    if (anpset.existsAs<std::string>(objStr + "_recCut", false)) {
      _recCut[it->first] = anpset.getUntrackedParameter<std::string>(objStr + "_recCut");
    }
    if (anpset.existsAs<double>(objStr + "_cutMinPt", false)) {
      _cutMinPt[it->first] = anpset.getUntrackedParameter<double>(objStr + "_cutMinPt");
    }
    if (anpset.existsAs<double>(objStr + "_cutMaxEta", false)) {
      _cutMaxEta[it->first] = anpset.getUntrackedParameter<double>(objStr + "_cutMaxEta");
    }
  }
  _hltPathsToCheck = anpset.getParameter<std::vector<std::string>>("hltPathsToCheck");
  _minCandidates = anpset.getParameter<unsigned int>("minCandidates");

  std::vector<double> default_parametersHt;
  default_parametersHt.push_back(100);
  default_parametersHt.push_back(0);
  default_parametersHt.push_back(1000);
  _parametersHt = pset.getUntrackedParameter<std::vector<double>>("parametersHt", default_parametersHt);

  _HtJetPtMin = anpset.getUntrackedParameter<double>("HtJetPtMin", -1);
  _HtJetEtaMax = anpset.getUntrackedParameter<double>("HtJetEtaMax", -1);

  if (_HtJetPtMin > 0 && _HtJetEtaMax > 0)
    _bookHtPlots = true;
  else
    _bookHtPlots = false;

  if (pset.exists("pileUpInfoLabel")) {
    _puSummaryInfo = iC.consumes<std::vector<PileupSummaryInfo>>(pset.getParameter<std::string>("pileUpInfoLabel"));
  }

  if (anpset.existsAs<std::vector<double>>("NminOneCuts", false)) {
    _NminOneCuts = anpset.getUntrackedParameter<std::vector<double>>("NminOneCuts");
    if (_NminOneCuts.size() < 9 + _minCandidates) {
      edm::LogError("HiggsValidation") << "In HLTHiggsSubAnalysis::HLTHiggsSubAnalysis, "
                                       << "Incoherence found in the python configuration file!!\nThe SubAnalysis '"
                                       << _analysisname << "' has a vector NminOneCuts with size "
                                       << _NminOneCuts.size() << ", while it needs to be at least of size "
                                       << (9 + _minCandidates) << ".";
      exit(-1);
    }
    if ((_NminOneCuts[0] || _NminOneCuts[1]) && _minCandidates < 4) {
      edm::LogError("HiggsValidation") << "In HLTHiggsSubAnalysis::HLTHiggsSubAnalysis, "
                                       << "Incoherence found in the python configuration file!!\nThe SubAnalysis '"
                                       << _analysisname
                                       << "' has a vector NminOneCuts with a dEtaqq of mqq cut on the least b-tagged "
                                          "jets of the first 4 jets while only requiring "
                                       << _minCandidates << " jets.";
      exit(-1);
    }
    if (_NminOneCuts[5] && _minCandidates < 3) {
      edm::LogError("HiggsValidation") << "In HLTHiggsSubAnalysis::HLTHiggsSubAnalysis, "
                                       << "Incoherence found in the python configuration file!!\nThe SubAnalysis '"
                                       << _analysisname
                                       << "' has a vector NminOneCuts with a CSV3 cut while only requiring "
                                       << _minCandidates << " jets.";
      exit(-1);
    }
    if ((_NminOneCuts[2] || _NminOneCuts[4]) && _minCandidates < 2) {
      edm::LogError("HiggsValidation") << "In HLTHiggsSubAnalysis::HLTHiggsSubAnalysis, "
                                       << "Incoherence found in the python configuration file!!\nThe SubAnalysis '"
                                       << _analysisname
                                       << "' has a vector NminOneCuts with a dPhibb or CSV2 cut using the second most "
                                          "b-tagged jet while only requiring "
                                       << _minCandidates << " jet.";
      exit(-1);
    }
    for (std::vector<double>::const_iterator it = _NminOneCuts.begin(); it != _NminOneCuts.end(); ++it) {
      if (*it) {
        _useNminOneCuts = true;
        break;
      }
    }
  }
  //    NptPlots = ( _useNminOneCuts ? _minCandidates : 2 );
  NptPlots = _minCandidates;
}

HLTHiggsSubAnalysis::~HLTHiggsSubAnalysis() {
  for (std::map<unsigned int, StringCutObjectSelector<reco::GenParticle>*>::iterator it = _genSelectorMap.begin();
       it != _genSelectorMap.end();
       ++it) {
    delete it->second;
    it->second = nullptr;
  }
  delete _genJetSelector;
  _genJetSelector = nullptr;
  delete _recMuonSelector;
  _recMuonSelector = nullptr;
  delete _recElecSelector;
  _recElecSelector = nullptr;
  delete _recPhotonSelector;
  _recPhotonSelector = nullptr;
  delete _recCaloMETSelector;
  _recCaloMETSelector = nullptr;
  delete _recPFMETSelector;
  _recPFMETSelector = nullptr;
  delete _recPFTauSelector;
  _recPFTauSelector = nullptr;
  delete _recPFJetSelector;
  _recPFJetSelector = nullptr;
  delete _recTrackSelector;
  _recTrackSelector = nullptr;
}

void HLTHiggsSubAnalysis::beginJob() {}

void HLTHiggsSubAnalysis::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  // Initialize the confighlt
  bool changedConfig;
  if (!_hltConfig.init(iRun, iSetup, _hltProcessName, changedConfig)) {
    edm::LogError("HiggsValidations") << "HLTHiggsSubAnalysis::beginRun: "
                                      << "Initializtion of HLTConfigProvider failed!!";
  }

  // Parse the input paths to get them if there are in the table
  // and associate them the last filter of the path (in order to extract the
  _hltPaths.clear();
  for (size_t i = 0; i < _hltPathsToCheck.size(); ++i) {
    bool found = false;
    TPRegexp pattern(_hltPathsToCheck[i]);
    for (size_t j = 0; j < _hltConfig.triggerNames().size(); ++j) {
      std::string thetriggername = _hltConfig.triggerNames()[j];
      if (TString(thetriggername).Contains(pattern)) {
        _hltPaths.insert(thetriggername);
        found = true;
      }
    }
    if (!found) {
      LogDebug("HiggsValidations") << "HLTHiggsSubAnalysis::beginRun, In " << _analysisname
                                   << " subfolder NOT found the path: '" << _hltPathsToCheck[i] << "*'";
    }
  }

  LogTrace("HiggsValidation") << "SubAnalysis: " << _analysisname << "\nHLT Trigger Paths found >>>";
  // Initialize the plotters (analysers for each trigger path)
  _analyzers.clear();
  for (std::set<std::string>::iterator iPath = _hltPaths.begin(); iPath != _hltPaths.end(); ++iPath) {
    // Avoiding the dependence of the version number for
    // the trigger paths
    std::string path = *iPath;
    std::string shortpath = path;
    if (path.rfind("_v") < path.length()) {
      shortpath = path.substr(0, path.rfind("_v"));
    }
    _shortpath2long[shortpath] = path;

    // Objects needed by the HLT path
    const std::vector<unsigned int> objsNeedHLT = this->getObjectsType(shortpath);
    // Sanity check: the object needed by a trigger path should be
    // introduced by the user via config python (_recLabels datamember)
    std::vector<unsigned int> userInstantiate;
    for (std::map<unsigned int, std::string>::iterator it = _recLabels.begin(); it != _recLabels.end(); ++it) {
      userInstantiate.push_back(it->first);
    }
    for (std::vector<unsigned int>::const_iterator it = objsNeedHLT.begin(); it != objsNeedHLT.end(); ++it) {
      if (std::find(userInstantiate.begin(), userInstantiate.end(), *it) == userInstantiate.end()) {
        edm::LogError("HiggsValidation") << "In HLTHiggsSubAnalysis::beginRun, "
                                         << "Incoherence found in the python configuration file!!\nThe SubAnalysis '"
                                         << _analysisname << "' has been asked to evaluate the trigger path '"
                                         << shortpath << "' (found it in 'hltPathsToCheck') BUT this path"
                                         << " needs a '" << EVTColContainer::getTypeString(*it)
                                         << "' which has not been instantiate ('recVariableLabels'"
                                         << ")";
        exit(-1);
      }
    }
    LogTrace("HiggsValidation") << " --- " << shortpath;

    // the hlt path, the objects (elec,muons,photons,...)
    // needed to evaluate the path are the argumens of the plotter
    HLTHiggsPlotter analyzer(_pset, shortpath, objsNeedHLT, NptPlots, _NminOneCuts);
    _analyzers.push_back(analyzer);
  }
}

void HLTHiggsSubAnalysis::bookHistograms(DQMStore::IBooker& ibooker) {
  std::string baseDir = _histDirectory + "/" + _analysisname + "/";
  ibooker.setCurrentFolder(baseDir);
  // Book the gen/reco analysis-dependent histograms (denominators)
  std::vector<std::string> sources(2);
  sources[0] = "gen";
  sources[1] = "rec";

  for (std::map<unsigned int, std::string>::const_iterator it = _recLabels.begin(); it != _recLabels.end(); ++it) {
    const std::string objStr = EVTColContainer::getTypeString(it->first);
    TString maxPt;

    for (size_t i = 0; i < sources.size(); i++) {
      std::string source = sources[i];
      if (_useNminOneCuts && it->first == EVTColContainer::PFJET) {
        if (source == "gen")
          continue;
        else {
          // N-1 jet plots (dEtaqq, mqq, dPhibb, CSV1, maxCSV_jets, maxCSV_E, PFMET, pt1, pt2, pt3, pt4)
          if (_NminOneCuts[0])
            bookHist(source, objStr, "dEtaqq", ibooker);
          if (_NminOneCuts[1])
            bookHist(source, objStr, "mqq", ibooker);
          if (_NminOneCuts[2])
            bookHist(source, objStr, "dPhibb", ibooker);
          if (_NminOneCuts[3]) {
            if (_NminOneCuts[6])
              bookHist(source, objStr, "maxCSV", ibooker);
            else
              bookHist(source, objStr, "CSV1", ibooker);
          }
          if (_NminOneCuts[4])
            bookHist(source, objStr, "CSV2", ibooker);
          if (_NminOneCuts[5])
            bookHist(source, objStr, "CSV3", ibooker);
        }
      }

      bookHist(source, objStr, "Eta", ibooker);
      bookHist(source, objStr, "Phi", ibooker);
      for (unsigned int i = 0; i < NptPlots; i++) {
        maxPt = "MaxPt";
        maxPt += i + 1;
        bookHist(source, objStr, maxPt.Data(), ibooker);
      }
    }
  }

  // Call the bookHistograms (which books all the path dependent histograms)
  for (std::vector<HLTHiggsPlotter>::iterator it = _analyzers.begin(); it != _analyzers.end(); ++it) {
    it->bookHistograms(ibooker, _useNminOneCuts);
  }
  //booking the histograms for overall trigger efficiencies
  for (size_t i = 0; i < sources.size(); i++) {
    std::string nameGlobalEfficiency = "SummaryPaths_" + _analysisname + "_" + sources[i];

    _elements[nameGlobalEfficiency] = ibooker.book1D(
        nameGlobalEfficiency.c_str(), nameGlobalEfficiency.c_str(), _hltPathsToCheck.size(), 0, _hltPathsToCheck.size());

    std::string nameGlobalEfficiencyPassing = nameGlobalEfficiency + "_passingHLT";
    _elements[nameGlobalEfficiencyPassing] = ibooker.book1D(nameGlobalEfficiencyPassing.c_str(),
                                                            nameGlobalEfficiencyPassing.c_str(),
                                                            _hltPathsToCheck.size(),
                                                            0,
                                                            _hltPathsToCheck.size());

    std::string titlePu = "nb of interations in the event";
    std::string nameVtxPlot = "trueVtxDist_" + _analysisname + "_" + sources[i];
    std::vector<double> paramsPu = _parametersPu;
    int nBinsPu = (int)paramsPu[0];
    double minPu = paramsPu[1];
    double maxPu = paramsPu[2];

    std::string titleHt = "sum of jet pT in the event";
    std::string nameHtPlot = "HtDist_" + _analysisname + "_" + sources[i];
    std::vector<double> paramsHt = _parametersHt;
    int nBinsHt = (int)paramsHt[0];
    double minHt = paramsHt[1];
    double maxHt = paramsHt[2];

    if ((!_useNminOneCuts) || sources[i] == "rec")
      _elements[nameVtxPlot] = ibooker.book1D(nameVtxPlot.c_str(), titlePu.c_str(), nBinsPu, minPu, maxPu);
    if (_bookHtPlots)
      _elements[nameHtPlot] = ibooker.book1D(nameHtPlot.c_str(), titleHt.c_str(), nBinsHt, minHt, maxHt);
    for (size_t j = 0; j < _hltPathsToCheck.size(); j++) {
      //declare the efficiency vs interaction plots
      std::string path = _hltPathsToCheck[j];
      std::string shortpath = path;
      if (path.rfind("_v") < path.length()) {
        shortpath = path.substr(0, path.rfind("_v"));
      }
      std::string titlePassingPu = "nb of interations in the event passing path " + shortpath;
      if ((!_useNminOneCuts) || sources[i] == "rec")
        _elements[nameVtxPlot + "_" + shortpath] =
            ibooker.book1D(nameVtxPlot + "_" + shortpath, titlePassingPu.c_str(), nBinsPu, minPu, maxPu);

      std::string titlePassingHt = "sum of jet pT in the event passing path " + shortpath;
      if (_bookHtPlots)
        _elements[nameHtPlot + "_" + shortpath] =
            ibooker.book1D(nameHtPlot + "_" + shortpath, titlePassingHt.c_str(), nBinsHt, minHt, maxHt);

      //fill the bin labels of the summary plot
      _elements[nameGlobalEfficiency]->setBinLabel(j + 1, shortpath);
      _elements[nameGlobalEfficiencyPassing]->setBinLabel(j + 1, shortpath);
    }
  }
}

void HLTHiggsSubAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, EVTColContainer* cols) {
  // Initialize the collection (the ones which hasn't been initialiazed yet)
  this->initobjects(iEvent, cols);
  // utility map
  std::map<unsigned int, std::string> u2str;
  u2str[GEN] = "gen";
  u2str[RECO] = "rec";

  std::map<unsigned int, double> Htmap;
  Htmap[GEN] = 0.;
  Htmap[RECO] = 0.;

  edm::Handle<std::vector<PileupSummaryInfo>> puInfo;
  iEvent.getByToken(_puSummaryInfo, puInfo);
  int nbMCvtx = -1;
  if (puInfo.isValid()) {
    std::vector<PileupSummaryInfo>::const_iterator PVI;
    for (PVI = puInfo->begin(); PVI != puInfo->end(); ++PVI) {
      if (PVI->getBunchCrossing() == 0) {
        nbMCvtx = PVI->getPU_NumInteractions();
        break;
      }
    }
  }

  // Extract the match structure containing the gen/reco candidates (electron, muons,...)
  // common to all the SubAnalysis
  //---- Generation
  // Make each good gen object into the base cand for a MatchStruct
  std::vector<MatchStruct>* matches = new std::vector<MatchStruct>;
  //  bool alreadyMu = false;
  for (std::map<unsigned int, std::string>::iterator it = _recLabels.begin(); it != _recLabels.end(); ++it) {
    // Use genJets for jet matchstructs
    if (it->first == EVTColContainer::PFJET) {
      // Skip genJets for N-1 plots
      if (!_useNminOneCuts) {
        // Initialize selectors when first event
        if (!_genJetSelector) {
          _genJetSelector = new StringCutObjectSelector<reco::GenJet>(_genCut[it->first]);
        }
        for (size_t i = 0; i < cols->genJets->size(); ++i) {
          if (_genJetSelector->operator()(cols->genJets->at(i))) {
            matches->push_back(MatchStruct(&cols->genJets->at(i), it->first));
          }
        }
      }
    }
    // Use genParticles
    else {
      // Avoiding the TkMu and Mu case
      /*      if( alreadyMu )
        {
            continue;
        }*/
      // Initialize selectors when first event
      if (!_genSelectorMap[it->first]) {
        _genSelectorMap[it->first] = new StringCutObjectSelector<reco::GenParticle>(_genCut[it->first]);
      }

      for (size_t i = 0; i < cols->genParticles->size(); ++i) {
        if (_genSelectorMap[it->first]->operator()(cols->genParticles->at(i))) {
          matches->push_back(MatchStruct(&cols->genParticles->at(i), it->first));
        }
      }
      /*      if( it->first == EVTColContainer::MUON || it->first == EVTColContainer::TRACK )
        {
            alreadyMu = true;
        }*/
    }
  }
  // Sort the MatchStructs by pT for later filling of turn-on curve
  std::sort(matches->begin(), matches->end(), matchesByDescendingPt());

  // Map to reference the source (gen/reco) with the recoCandidates
  std::map<unsigned int, std::vector<MatchStruct>> sourceMatchMap;  // To be a pointer to delete
  // --- Storing the generating candidates
  sourceMatchMap[GEN] = *matches;

  // Reuse the vector
  matches->clear();
  // --- same for RECO objects

  // Different treatment for jets (b-tag)
  std::map<std::string, bool> nMinOne;
  std::map<std::string, bool> jetCutResult;
  float dEtaqq;
  float mqq;
  float dPhibb;
  float CSV1;
  float CSV2;
  float CSV3;
  bool passAllCuts = false;
  if (_recLabels.find(EVTColContainer::PFJET) != _recLabels.end()) {
    // Initialize jet selector
    this->InitSelector(EVTColContainer::PFJET);
    // Initialize and insert pfJets
    this->initAndInsertJets(iEvent, cols, matches);
    // Make sure to skip events that don't have enough jets
    if (matches->size() < NptPlots) {
      delete matches;
      return;
    }
    // Cuts on multiple jet events (RECO)
    if (_useNminOneCuts) {
      this->passJetCuts(matches, jetCutResult, dEtaqq, mqq, dPhibb, CSV1, CSV2, CSV3);
    }
  }
  // Extraction of the objects candidates
  for (std::map<unsigned int, std::string>::iterator it = _recLabels.begin(); it != _recLabels.end(); ++it) {
    // Reco selectors (the function takes into account if it was instantiated
    // before or not
    this->InitSelector(it->first);
    // -- Storing the matches
    this->insertcandidates(it->first, cols, matches);
  }

  // Sort the MatchStructs by pT for later filling of turn-on curve
  std::sort(matches->begin(), matches->end(), matchesByDescendingPt());

  if (_useNminOneCuts) {
    // Check non-jet N-1 Cuts
    this->passOtherCuts(*matches, jetCutResult);

    // Make N-1 booleans from jetCutResults
    for (std::map<std::string, bool>::const_iterator it = jetCutResult.begin(); it != jetCutResult.end(); ++it) {
      nMinOne[it->first] = true;
      for (std::map<std::string, bool>::const_iterator it2 = jetCutResult.begin(); it2 != jetCutResult.end(); ++it2) {
        //ignore CSV2,CSV3 cut plotting CSV1
        if (it->first == "CSV1" && it2->first == "CSV3")
          continue;
        if (it->first == "CSV1" && it2->first == "CSV2")
          continue;

        //ignore CSV3 plotting cut CSV2
        if (it->first == "CSV2" && it2->first == "CSV3")
          continue;

        if (it->first != it2->first && !(it2->second)) {
          nMinOne[it->first] = false;
          break;
        }
      }
    }
    bool temp = false;
    for (std::map<std::string, bool>::const_iterator it = nMinOne.begin(); it != nMinOne.end(); ++it) {
      if (temp && it->second) {
        passAllCuts = true;
        break;
      }
      if (it->second)
        temp = true;
    }
  }

  // --- Storing the reco candidates
  sourceMatchMap[RECO] = *matches;
  // --- All the objects are in place
  delete matches;

  // -- Trigger Results
  const edm::TriggerNames& trigNames = iEvent.triggerNames(*(cols->triggerResults));

  if (_bookHtPlots) {
    edm::Handle<reco::PFJetCollection> recoJet;
    iEvent.getByToken(_recoHtJetLabel, recoJet);
    if (recoJet.isValid()) {
      for (reco::PFJetCollection::const_iterator iJet = recoJet->begin(); iJet != recoJet->end(); iJet++) {
        double pt = iJet->pt();
        double eta = iJet->eta();
        if (pt > _HtJetPtMin && fabs(eta) < _HtJetEtaMax) {
          Htmap[RECO] += pt;
        }
      }
    }

    edm::Handle<reco::GenJetCollection> genJet;
    iEvent.getByToken(_genJetLabel, genJet);
    if (genJet.isValid()) {
      for (reco::GenJetCollection::const_iterator iJet = genJet->begin(); iJet != genJet->end(); iJet++) {
        double pt = iJet->pt();
        double eta = iJet->eta();
        if (pt > _HtJetPtMin && fabs(eta) < _HtJetEtaMax) {
          Htmap[GEN] += pt;
        }
      }
    }
  }

  // Filling the histograms if pass the minimum amount of candidates needed by the analysis:
  // GEN + RECO CASE in the same loop
  for (std::map<unsigned int, std::vector<MatchStruct>>::iterator it = sourceMatchMap.begin();
       it != sourceMatchMap.end();
       ++it) {
    // it->first: gen/reco   it->second: matches (std::vector<MatchStruc>)
    if (it->second.size() < _minCandidates)  // FIXME: A bug is potentially here: what about the mixed channels?
    {
      continue;
    }

    // Filling the gen/reco objects (eff-denominators):
    // Just the first two different ones, if there are more
    std::map<unsigned int, int>* countobjects = new std::map<unsigned int, int>;
    // Initializing the count of the used object
    for (std::map<unsigned int, std::string>::iterator co = _recLabels.begin(); co != _recLabels.end(); ++co) {
      if (!(_useNminOneCuts && co->first == EVTColContainer::PFJET && it->first == GEN))  // genJets are not there
        countobjects->insert(std::pair<unsigned int, int>(co->first, 0));
    }
    int counttotal = 0;
    const int totalobjectssize2 = NptPlots * countobjects->size();
    for (size_t j = 0; j < it->second.size(); ++j) {
      const unsigned int objType = it->second[j].objType;
      const std::string objTypeStr = EVTColContainer::getTypeString(objType);

      float pt = (it->second)[j].pt;
      float eta = (it->second)[j].eta;
      float phi = (it->second)[j].phi;

      // PFMET N-1 cut
      if (_useNminOneCuts && objType == EVTColContainer::PFMET && _NminOneCuts[8] && !nMinOne["PFMET"]) {
        continue;
      }

      TString maxPt;
      if ((unsigned)(*countobjects)[objType] < NptPlots) {
        maxPt = "MaxPt";
        maxPt += (*countobjects)[objType] + 1;
        if (_useNminOneCuts && objType == EVTColContainer::PFJET) {
          if (nMinOne[maxPt.Data()]) {
            this->fillHist(u2str[it->first], objTypeStr, maxPt.Data(), pt);
          }
        } else {
          this->fillHist(u2str[it->first], objTypeStr, maxPt.Data(), pt);
        }
        // Filled the high pt ...
        ++((*countobjects)[objType]);
        ++counttotal;
      } else {
        if ((unsigned)(*countobjects)[objType] < _minCandidates) {  // To get correct results for HZZ
          ++((*countobjects)[objType]);
          ++counttotal;
        } else
          continue;  //   Otherwise too many entries in Eta and Phi distributions
      }

      // Jet N-1 Cuts
      if (_useNminOneCuts && objType == EVTColContainer::PFJET) {
        if (passAllCuts) {
          this->fillHist(u2str[it->first], objTypeStr, "Eta", eta);
          this->fillHist(u2str[it->first], objTypeStr, "Phi", phi);
        }
      } else {
        this->fillHist(u2str[it->first], objTypeStr, "Eta", eta);
        this->fillHist(u2str[it->first], objTypeStr, "Phi", phi);
      }

      // Already the minimum two objects has been filled, get out...
      if (counttotal == totalobjectssize2) {
        break;
      }
    }
    delete countobjects;

    if (_useNminOneCuts && it->first == RECO) {
      if (_NminOneCuts[0] && nMinOne["dEtaqq"]) {
        this->fillHist(u2str[it->first], EVTColContainer::getTypeString(EVTColContainer::PFJET), "dEtaqq", dEtaqq);
      }
      if (_NminOneCuts[1] && nMinOne["mqq"]) {
        this->fillHist(u2str[it->first], EVTColContainer::getTypeString(EVTColContainer::PFJET), "mqq", mqq);
      }
      if (_NminOneCuts[2] && nMinOne["dPhibb"]) {
        this->fillHist(u2str[it->first], EVTColContainer::getTypeString(EVTColContainer::PFJET), "dPhibb", dPhibb);
      }
      if (_NminOneCuts[3]) {
        std::string nameCSVplot = "CSV1";
        if (_NminOneCuts[6])
          nameCSVplot = "maxCSV";
        if (nMinOne[nameCSVplot])
          this->fillHist(u2str[it->first], EVTColContainer::getTypeString(EVTColContainer::PFJET), nameCSVplot, CSV1);
      }
      if (_NminOneCuts[4] && nMinOne["CSV2"]) {
        this->fillHist(u2str[it->first], EVTColContainer::getTypeString(EVTColContainer::PFJET), "CSV2", CSV2);
      }
      if (_NminOneCuts[5] && nMinOne["CSV3"]) {
        this->fillHist(u2str[it->first], EVTColContainer::getTypeString(EVTColContainer::PFJET), "CSV3", CSV3);
      }
    }

    //fill the efficiency vs nb of interactions
    std::string nameVtxPlot = "trueVtxDist_" + _analysisname + "_" + u2str[it->first];
    if ((!_useNminOneCuts) || it->first == RECO)
      _elements[nameVtxPlot]->Fill(nbMCvtx);

    //fill the efficiency vs sum pT of jets
    std::string nameHtPlot = "HtDist_" + _analysisname + "_" + u2str[it->first];
    if (_bookHtPlots)
      _elements[nameHtPlot]->Fill(Htmap[it->first]);

    // Calling to the plotters analysis (where the evaluation of the different trigger paths are done)
    std::string SummaryName = "SummaryPaths_" + _analysisname + "_" + u2str[it->first];
    const std::string source = u2str[it->first];
    for (std::vector<HLTHiggsPlotter>::iterator an = _analyzers.begin(); an != _analyzers.end(); ++an) {
      const std::string hltPath = _shortpath2long[an->gethltpath()];
      const std::string fillShortPath = an->gethltpath();
      const bool ispassTrigger = cols->triggerResults->accept(trigNames.triggerIndex(hltPath));

      if (_useNminOneCuts) {
        an->analyze(ispassTrigger, source, it->second, nMinOne, dEtaqq, mqq, dPhibb, CSV1, CSV2, CSV3, passAllCuts);
      } else {
        an->analyze(ispassTrigger, source, it->second, _minCandidates);
      }

      int refOfThePath = -1;
      for (size_t itePath = 0; itePath < _hltPathsToCheck.size(); itePath++) {
        refOfThePath++;
        if (TString(hltPath).Contains(_hltPathsToCheck[itePath]))
          break;
      }
      _elements[SummaryName]->Fill(refOfThePath);
      if (ispassTrigger) {
        _elements[SummaryName + "_passingHLT"]->Fill(refOfThePath, 1);
        if ((!_useNminOneCuts) || it->first == RECO)
          _elements[nameVtxPlot + "_" + fillShortPath]->Fill(nbMCvtx);
        if (_bookHtPlots)
          _elements[nameHtPlot + "_" + fillShortPath]->Fill(Htmap[it->first]);
      } else {
        _elements[SummaryName + "_passingHLT"]->Fill(refOfThePath, 0);
      }
    }
  }
}

// Return the objects (muons,electrons,photons,...) needed by a hlt path.
const std::vector<unsigned int> HLTHiggsSubAnalysis::getObjectsType(const std::string& hltPath) const {
  static const unsigned int objSize = 7;
  static const unsigned int objtriggernames[] = {EVTColContainer::MUON,
                                                 EVTColContainer::ELEC,
                                                 EVTColContainer::PHOTON,
                                                 //      EVTColContainer::TRACK,  // Note is tracker muon
                                                 EVTColContainer::PFTAU,
                                                 EVTColContainer::PFJET,
                                                 EVTColContainer::CALOMET,
                                                 EVTColContainer::PFMET};

  std::set<unsigned int> objsType;
  // The object to deal has to be entered via the config .py
  for (unsigned int i = 0; i < objSize; ++i) {
    std::string objTypeStr = EVTColContainer::getTypeString(objtriggernames[i]);
    // Check if it is needed this object for this trigger
    if (!TString(hltPath).Contains(objTypeStr)) {
      if ((objtriggernames[i] == EVTColContainer::PFJET &&
           TString(hltPath).Contains("WHbbBoost")) ||  // fix for HLT_Ele27_WPLoose_Gsf_WHbbBoost_v
          (objtriggernames[i] == EVTColContainer::PFJET && TString(hltPath).Contains("CSV")) ||  // fix for ZnnHbb PFJET
          (objtriggernames[i] == EVTColContainer::PFMET && TString(hltPath).Contains("MHT")) ||  // fix for ZnnHbb PFMET
          (objtriggernames[i] == EVTColContainer::PHOTON && TString(hltPath).Contains("Diphoton"))) {
        objsType.insert(objtriggernames[i]);  //case of the New Diphoton paths
      }
      continue;
    }
    if ((objtriggernames[i] == EVTColContainer::CALOMET &&
         (TString(hltPath).Contains("PFMET") || TString(hltPath).Contains("MHT"))) ||  // fix for PFMET
        (objtriggernames[i] == EVTColContainer::PFJET && TString(hltPath).Contains("JetIdCleaned") &&
         !TString(hltPath).Contains(TRegexp("Jet[^I]"))) ||                                     // fix for Htaunu
        (objtriggernames[i] == EVTColContainer::MUON && TString(hltPath).Contains("METNoMu")))  // fix for VBFHToInv
    {
      continue;
    }

    objsType.insert(objtriggernames[i]);
  }

  return std::vector<unsigned int>(objsType.begin(), objsType.end());
}

// Booking the maps: recLabels and genParticle selectors
void HLTHiggsSubAnalysis::bookobjects(const edm::ParameterSet& anpset, edm::ConsumesCollector& iC) {
  if (anpset.exists("recMuonLabel")) {
    _recLabels[EVTColContainer::MUON] = anpset.getParameter<std::string>("recMuonLabel");
    _recLabelsMuon = iC.consumes<reco::MuonCollection>(anpset.getParameter<std::string>("recMuonLabel"));
    _genSelectorMap[EVTColContainer::MUON] = nullptr;
  }
  if (anpset.exists("recElecLabel")) {
    _recLabels[EVTColContainer::ELEC] = anpset.getParameter<std::string>("recElecLabel");
    _recLabelsElec = iC.consumes<reco::GsfElectronCollection>(anpset.getParameter<std::string>("recElecLabel"));
    _genSelectorMap[EVTColContainer::ELEC] = nullptr;
  }
  if (anpset.exists("recPhotonLabel")) {
    _recLabels[EVTColContainer::PHOTON] = anpset.getParameter<std::string>("recPhotonLabel");
    _recLabelsPhoton = iC.consumes<reco::PhotonCollection>(anpset.getParameter<std::string>("recPhotonLabel"));
    _genSelectorMap[EVTColContainer::PHOTON] = nullptr;
  }
  if (anpset.exists("recCaloMETLabel")) {
    _recLabels[EVTColContainer::CALOMET] = anpset.getParameter<std::string>("recCaloMETLabel");
    _recLabelsCaloMET = iC.consumes<reco::CaloMETCollection>(anpset.getParameter<std::string>("recCaloMETLabel"));
    _genSelectorMap[EVTColContainer::CALOMET] = nullptr;
  }
  if (anpset.exists("recPFMETLabel")) {
    _recLabels[EVTColContainer::PFMET] = anpset.getParameter<std::string>("recPFMETLabel");
    _recLabelsPFMET = iC.consumes<reco::PFMETCollection>(anpset.getParameter<std::string>("recPFMETLabel"));
    _genSelectorMap[EVTColContainer::PFMET] = nullptr;
  }
  if (anpset.exists("recPFTauLabel")) {
    _recLabels[EVTColContainer::PFTAU] = anpset.getParameter<std::string>("recPFTauLabel");
    _recLabelsPFTau = iC.consumes<reco::PFTauCollection>(anpset.getParameter<std::string>("recPFTauLabel"));
    _genSelectorMap[EVTColContainer::PFTAU] = nullptr;
  }
  if (anpset.exists("recJetLabel")) {
    _recLabels[EVTColContainer::PFJET] = anpset.getParameter<std::string>("recJetLabel");
    _recLabelsPFJet = iC.consumes<reco::PFJetCollection>(anpset.getParameter<std::string>("recJetLabel"));
    if (anpset.exists("jetTagLabel"))
      _recTagPFJet = iC.consumes<reco::JetTagCollection>(anpset.getParameter<std::string>("jetTagLabel"));
    _genJetSelector = nullptr;
  }
  /*if( anpset.exists("recTrackLabel") )
    {
        _recLabels[EVTColContainer::TRACK] = anpset.getParameter<std::string>("recTrackLabel");
        _genSelectorMap[EVTColContainer::TRACK] = 0 ;
    }*/

  if (_recLabels.empty()) {
    edm::LogError("HiggsValidation") << "HLTHiggsSubAnalysis::bookobjects, "
                                     << "Not included any object (recMuonLabel, recElecLabel, ...)  "
                                     << "in the analysis " << _analysisname;
    return;
  }
}

void HLTHiggsSubAnalysis::initobjects(const edm::Event& iEvent, EVTColContainer* col) {
  /*if( col != 0 && col->isAllInit() )
    {
        // Already init, not needed to do nothing
        return;
    }*/
  if (!col->isCommonInit()) {
    // extract the trigger results (path info, pass,...)
    edm::Handle<edm::TriggerResults> trigResults;
    iEvent.getByToken(_trigResultsTag, trigResults);
    if (trigResults.isValid()) {
      col->triggerResults = trigResults.product();
    }

    // GenParticle collection if is there (genJets only if there need to be jets)
    edm::Handle<reco::GenParticleCollection> genPart;
    iEvent.getByToken(_genParticleLabel, genPart);
    if (genPart.isValid()) {
      col->genParticles = genPart.product();
    }
  }

  for (std::map<unsigned int, std::string>::iterator it = _recLabels.begin(); it != _recLabels.end(); ++it) {
    if (it->first == EVTColContainer::MUON) {
      edm::Handle<reco::MuonCollection> theHandle;
      iEvent.getByToken(_recLabelsMuon, theHandle);
      col->set(theHandle.product());
    } else if (it->first == EVTColContainer::ELEC) {
      edm::Handle<reco::GsfElectronCollection> theHandle;
      iEvent.getByToken(_recLabelsElec, theHandle);
      col->set(theHandle.product());
    } else if (it->first == EVTColContainer::PHOTON) {
      edm::Handle<reco::PhotonCollection> theHandle;
      iEvent.getByToken(_recLabelsPhoton, theHandle);
      col->set(theHandle.product());
    } else if (it->first == EVTColContainer::CALOMET) {
      edm::Handle<reco::CaloMETCollection> theHandle;
      iEvent.getByToken(_recLabelsCaloMET, theHandle);
      col->set(theHandle.product());
    } else if (it->first == EVTColContainer::PFMET) {
      edm::Handle<reco::PFMETCollection> theHandle;
      iEvent.getByToken(_recLabelsPFMET, theHandle);
      col->set(theHandle.product());
    } else if (it->first == EVTColContainer::PFTAU) {
      edm::Handle<reco::PFTauCollection> theHandle;
      iEvent.getByToken(_recLabelsPFTau, theHandle);
      col->set(theHandle.product());
    }
    // PFJets loaded in seperate function initAndInsertJets because they need to be combined with the btags using the Handle (not the product) and for ordering them seperately in the MatchStruct's
    else if (it->first == EVTColContainer::PFJET) {
      if (!_useNminOneCuts) {
        // GenJet collection
        edm::Handle<reco::GenJetCollection> genJet;
        iEvent.getByToken(_genJetLabel, genJet);
        if (genJet.isValid()) {
          col->genJets = genJet.product();
        }
      }
    } else {
      edm::LogError("HiggsValidation") << "HLTHiggsSubAnalysis::initobjects "
                                       << " NOT IMPLEMENTED (yet) ERROR: '" << it->second << "'";
      //return; ??
    }
  }
}

void HLTHiggsSubAnalysis::bookHist(const std::string& source,
                                   const std::string& objType,
                                   const std::string& variable,
                                   DQMStore::IBooker& ibooker) {
  std::string sourceUpper = source;
  sourceUpper[0] = std::toupper(sourceUpper[0]);
  std::string name = source + objType + variable;
  TH1F* h = nullptr;

  if (variable.find("MaxPt") != std::string::npos) {
    std::string desc;
    if (variable == "MaxPt1")
      desc = "Leading";
    else if (variable == "MaxPt2")
      desc = "Next-to-Leading";
    else
      desc = variable.substr(5, 6) + "th Leading";
    std::string title = "pT of " + desc + " " + sourceUpper + " " + objType;
    const size_t nBinsStandard = _parametersTurnOn.size() - 1;
    size_t nBins = nBinsStandard;
    float* edges = new float[nBinsStandard + 1];
    for (size_t i = 0; i < nBinsStandard + 1; i++) {
      edges[i] = _parametersTurnOn[i];
    }

    std::string jetObj = EVTColContainer::getTypeString(EVTColContainer::PFJET);
    if (objType == jetObj) {
      const size_t nBinsJets = 25;
      nBins = nBinsJets;
      delete[] edges;
      edges = new float[nBinsJets + 1];
      for (size_t i = 0; i < nBinsJets + 1; i++) {
        edges[i] = i * 10;
      }
    }
    if (objType == EVTColContainer::getTypeString(EVTColContainer::PFMET)) {
      const size_t nBinsJets = 30;
      nBins = nBinsJets;
      delete[] edges;
      edges = new float[nBinsJets + 1];
      for (size_t i = 0; i < nBinsJets + 1; i++) {
        edges[i] = i * 10;
      }
    }
    h = new TH1F(name.c_str(), title.c_str(), nBins, edges);
    delete[] edges;
  } else if (variable == "dEtaqq") {
    std::string title = "#Delta #eta_{qq} of " + sourceUpper + " " + objType;
    int nBins = 20;
    double min = 0;
    double max = 4.8;
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  } else if (variable == "mqq") {
    std::string title = "m_{qq} of " + sourceUpper + " " + objType;
    int nBins = 20;
    double min = 0;
    double max = 1000;
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  } else if (variable == "dPhibb") {
    std::string title = "#Delta #phi_{bb} of " + sourceUpper + " " + objType;
    int nBins = 10;
    double min = 0;
    double max = 3.1416;
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  } else if (variable == "CSV1") {
    std::string title = "CSV1 of " + sourceUpper + " " + objType;
    int nBins = 20;
    double min = 0;
    double max = 1;
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  } else if (variable == "CSV2") {
    std::string title = "CSV2 of " + sourceUpper + " " + objType;
    int nBins = 20;
    double min = 0;
    double max = 1;
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  } else if (variable == "CSV3") {
    std::string title = "CSV3 of " + sourceUpper + " " + objType;
    int nBins = 20;
    double min = 0;
    double max = 1;
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  } else if (variable == "maxCSV") {
    std::string title = "max CSV of " + sourceUpper + " " + objType;
    int nBins = 20;
    double min = 0;
    double max = 1;
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  } else {
    std::string symbol = (variable == "Eta") ? "#eta" : "#phi";
    std::string title = symbol + " of " + sourceUpper + " " + objType;
    std::vector<double> params = (variable == "Eta") ? _parametersEta : _parametersPhi;

    int nBins = (int)params[0];
    double min = params[1];
    double max = params[2];
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  }
  h->Sumw2();
  _elements[name] = ibooker.book1D(name, h);
  delete h;
}

void HLTHiggsSubAnalysis::fillHist(const std::string& source,
                                   const std::string& objType,
                                   const std::string& variable,
                                   const float& value) {
  std::string sourceUpper = source;
  sourceUpper[0] = toupper(sourceUpper[0]);
  std::string name = source + objType + variable;

  _elements[name]->Fill(value);
}

// Initialize the selectors
void HLTHiggsSubAnalysis::InitSelector(const unsigned int& objtype) {
  if (objtype == EVTColContainer::MUON && _recMuonSelector == nullptr) {
    _recMuonSelector = new StringCutObjectSelector<reco::Muon>(_recCut[objtype]);
  } else if (objtype == EVTColContainer::ELEC && _recElecSelector == nullptr) {
    _recElecSelector = new StringCutObjectSelector<reco::GsfElectron>(_recCut[objtype]);
  } else if (objtype == EVTColContainer::PHOTON && _recPhotonSelector == nullptr) {
    _recPhotonSelector = new StringCutObjectSelector<reco::Photon>(_recCut[objtype]);
  } else if (objtype == EVTColContainer::CALOMET && _recCaloMETSelector == nullptr) {
    _recCaloMETSelector = new StringCutObjectSelector<reco::CaloMET>(_recCut[objtype]);
  } else if (objtype == EVTColContainer::PFMET && _recPFMETSelector == nullptr) {
    _recPFMETSelector = new StringCutObjectSelector<reco::PFMET>(_recCut[objtype]);
  } else if (objtype == EVTColContainer::PFTAU && _recPFTauSelector == nullptr) {
    _recPFTauSelector = new StringCutObjectSelector<reco::PFTau>(_recCut[objtype]);
  } else if (objtype == EVTColContainer::PFJET && _recPFJetSelector == nullptr) {
    _recPFJetSelector = new StringCutObjectSelector<reco::PFJet>(_recCut[objtype]);
  }
  /*else if( objtype == EVTColContainer::TRACK && _recTrackSelector == 0)
    {
        _recTrackSelector = new StringCutObjectSelector<reco::Track>(_recCut[objtype]);
    }*/
  /*  else
    {
FIXME: ERROR NO IMPLEMENTADO
    }*/
}

void HLTHiggsSubAnalysis::initAndInsertJets(const edm::Event& iEvent,
                                            EVTColContainer* cols,
                                            std::vector<MatchStruct>* matches) {
  edm::Handle<reco::PFJetCollection> PFJetHandle;
  iEvent.getByToken(_recLabelsPFJet, PFJetHandle);
  cols->set(PFJetHandle.product());

  edm::Handle<reco::JetTagCollection> JetTagHandle;
  if (_useNminOneCuts) {
    iEvent.getByToken(_recTagPFJet, JetTagHandle);
  }

  for (reco::PFJetCollection::const_iterator it = PFJetHandle->begin(); it != PFJetHandle->end(); ++it) {
    reco::PFJetRef jetRef(PFJetHandle, it - PFJetHandle->begin());
    reco::JetBaseRef jetBaseRef(jetRef);

    if (_recPFJetSelector->operator()(*it)) {
      if (_useNminOneCuts) {
        float bTag = (*(JetTagHandle.product()))[jetBaseRef];
        matches->push_back(MatchStruct(&*it, EVTColContainer::PFJET, bTag));
      } else {
        matches->push_back(MatchStruct(&*it, EVTColContainer::PFJET));
      }
    }
  }
}

void HLTHiggsSubAnalysis::passJetCuts(
    std::vector<MatchStruct>* matches,
    std::map<std::string, bool>& jetCutResult,
    float& dEtaqq,
    float& mqq,
    float& dPhibb,
    float& CSV1,
    float& CSV2,
    float& CSV3) {  //dEtaqq, mqq, dPhibb, CSV1, CSV2, CSV3, maxCSV_jets, maxCSV_E, PFMET, pt1, pt2, pt3, pt4

  // Perform pt cuts
  std::sort(matches->begin(), matches->end(), matchesByDescendingPt());
  TString maxPt;
  for (unsigned int i = 0; i < NptPlots; i++) {
    maxPt = "MaxPt";
    maxPt += i + 1;
    if ((*matches)[i].pt > _NminOneCuts[9 + i])
      jetCutResult[maxPt.Data()] = true;
    else
      jetCutResult[maxPt.Data()] = false;
  }

  unsigned int NbTag = ((_NminOneCuts[0] || _NminOneCuts[1]) ? 4 : 8);
  if (matches->size() < NbTag)
    NbTag = matches->size();
  // Perform b-tag ordered cuts
  std::sort(matches->begin(), matches->begin() + NbTag, matchesByDescendingBtag());

  if (_NminOneCuts[0]) {
    jetCutResult["dEtaqq"] = false;
    if (matches->size() > 2) {
      dEtaqq = fabs((*matches)[2].eta - (*matches)[3].eta);
      if (dEtaqq > _NminOneCuts[0])
        jetCutResult["dEtaqq"] = true;
    }
  }

  if (_NminOneCuts[1]) {
    jetCutResult["mqq"] = false;
    if (matches->size() > 2) {
      mqq = ((*matches)[2].lorentzVector + (*matches)[3].lorentzVector).M();
      if (mqq > _NminOneCuts[1])
        jetCutResult["mqq"] = true;
    }
  }

  if (_NminOneCuts[2]) {
    jetCutResult["dPhibb"] = false;
    if (matches->size() > 1) {
      dPhibb = fmod(fabs((*matches)[0].phi - (*matches)[1].phi), 3.1416);
      if (dPhibb < _NminOneCuts[2])
        jetCutResult["dPhibb"] = true;
    }
  }

  if (_NminOneCuts[4]) {
    std::string nameCSV2plot = "CSV2";
    jetCutResult[nameCSV2plot] = false;
    if (matches->size() > 1) {
      CSV2 = (*matches)[1].bTag;
      if (CSV2 > _NminOneCuts[4])
        jetCutResult[nameCSV2plot] = true;
    }
  }

  if (_NminOneCuts[5]) {
    std::string nameCSV3plot = "CSV3";
    jetCutResult[nameCSV3plot] = false;
    if (matches->size() > 2) {
      CSV3 = (*matches)[2].bTag;
      if (CSV3 > _NminOneCuts[5])
        jetCutResult[nameCSV3plot] = true;
    }
  }

  if (_NminOneCuts[3]) {
    CSV1 = (*matches)[0].bTag;
    std::string nameCSVplot = "CSV1";
    if (_NminOneCuts[6])
      nameCSVplot = "maxCSV";

    if (CSV1 > _NminOneCuts[3])
      jetCutResult[nameCSVplot] = true;
    else
      jetCutResult[nameCSVplot] = false;

    // max(CSV)
    if (_NminOneCuts[6]) {
      std::sort(matches->begin(), matches->end(), matchesByDescendingPt());
      CSV1 = (*matches)[0].bTag;
      unsigned int Njets = (unsigned int)_NminOneCuts[6];
      if (_NminOneCuts[6] > matches->size())
        Njets = matches->size();
      for (unsigned int i = 1; i < (unsigned int)Njets; ++i) {
        if ((*matches)[i].bTag > CSV1 && (*matches)[i].pt > _NminOneCuts[7])
          CSV1 = (*matches)[i].bTag;
      }
    }
  }
}

void HLTHiggsSubAnalysis::passOtherCuts(const std::vector<MatchStruct>& matches,
                                        std::map<std::string, bool>& jetCutResult) {
  if (_NminOneCuts[8]) {
    jetCutResult["PFMET"] = false;
    for (std::vector<MatchStruct>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
      if (it->objType == EVTColContainer::PFMET) {
        if (it->pt > _NminOneCuts[8])
          jetCutResult["PFMET"] = true;
        break;
      }
    }
  }
}

void HLTHiggsSubAnalysis::insertcandidates(const unsigned int& objType,
                                           const EVTColContainer* cols,
                                           std::vector<MatchStruct>* matches) {
  if (objType == EVTColContainer::MUON) {
    for (size_t i = 0; i < cols->muons->size(); i++) {
      if (_recMuonSelector->operator()(cols->muons->at(i))) {
        matches->push_back(MatchStruct(&cols->muons->at(i), objType));
      }
    }
  } else if (objType == EVTColContainer::ELEC) {
    for (size_t i = 0; i < cols->electrons->size(); i++) {
      if (_recElecSelector->operator()(cols->electrons->at(i))) {
        matches->push_back(MatchStruct(&cols->electrons->at(i), objType));
      }
    }
  } else if (objType == EVTColContainer::PHOTON) {
    for (size_t i = 0; i < cols->photons->size(); i++) {
      if (_recPhotonSelector->operator()(cols->photons->at(i))) {
        matches->push_back(MatchStruct(&cols->photons->at(i), objType));
      }
    }
  } else if (objType == EVTColContainer::CALOMET) {
    for (size_t i = 0; i < cols->caloMETs->size(); i++) {
      if (_recCaloMETSelector->operator()(cols->caloMETs->at(i))) {
        matches->push_back(MatchStruct(&cols->caloMETs->at(i), objType));
      }
    }
  } else if (objType == EVTColContainer::PFMET) {
    for (size_t i = 0; i < cols->pfMETs->size(); i++) {
      if (_recPFMETSelector->operator()(cols->pfMETs->at(i))) {
        matches->push_back(MatchStruct(&cols->pfMETs->at(i), objType));
      }
    }
  } else if (objType == EVTColContainer::PFTAU) {
    for (size_t i = 0; i < cols->pfTaus->size(); i++) {
      if (_recPFTauSelector->operator()(cols->pfTaus->at(i))) {
        matches->push_back(MatchStruct(&cols->pfTaus->at(i), objType));
      }
    }
  }
  //  else if( objType == EVTColContainer::PFJET )
  //  {
  //  already inserted
  //  }
  /*else if( objType == EVTColContainer::TRACK )
    {
        for(size_t i = 0; i < cols->tracks->size(); i++)
        {
            if(_recTrackSelector->operator()(cols->tracks->at(i)))
            {
                matches->push_back(MatchStruct(&cols->tracks->at(i),objType));
            }
        }
    }*/
  /*
    else FIXME: Control errores
    {
    }
    */
}
