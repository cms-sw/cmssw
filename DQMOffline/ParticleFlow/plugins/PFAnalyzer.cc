/** \class PFAnalyzer
 *
 *  DQM ParticleFlow analysis monitoring
 *
 *  \author J. Roloff - Brown University
 *
 */

#include "DQMOffline/ParticleFlow/plugins/PFAnalyzer.h"
#include <iostream>

// ***********************************************************
PFAnalyzer::PFAnalyzer(const edm::ParameterSet& pSet) {
  m_directory = "ParticleFlow";
  m_isMiniAOD = true;
  parameters_ = pSet.getParameter<edm::ParameterSet>("pfAnalysis");

  //patPfCandidateCollection_ = consumes<pat::PFParticleCollection>(pSet.getParameter<edm::InputTag>("pfCandidates"));
  patPfCandidateCollection_ =
      consumes<pat::PackedCandidateCollection>(pSet.getParameter<edm::InputTag>("pfCandidates"));
  patJetsToken_ = consumes<pat::JetCollection>(pSet.getParameter<edm::InputTag>("pfJetCollection"));

  thePfCandidateCollection_ = consumes<reco::PFCandidateCollection>(pSet.getParameter<edm::InputTag>("pfCandidates"));
  pfJetsToken_ = consumes<reco::PFJetCollection>(pSet.getParameter<edm::InputTag>("pfJetCollection"));

  theTriggerResultsLabel_ = pSet.getParameter<edm::InputTag>("TriggerResultsLabel");
  m_selection = pSet.getParameter<std::string>("eventSelection");

  triggerResultsToken_ = consumes<edm::TriggerResults>(edm::InputTag(theTriggerResultsLabel_));
  m_triggerOptions = pSet.getParameter<vstring>("TriggerNames");

  //puppiWeightToken_ = consumes<edm::ValueMap<float>>(pSet.getParameter<edm::InputTag>("puppiWeight"));

  m_pfNames = {"allPFC", "neutralHadPFC", "chargedHadPFC", "electronPFC", "muonPFC", "gammaPFC", "hadHFPFC", "emHFPFC"};
  vertexTag_ = pSet.getParameter<edm::InputTag>("PVCollection");
  vertexToken_ = consumes<std::vector<reco::Vertex>>(edm::InputTag(vertexTag_));

  tok_ew_ = consumes<GenEventInfoProduct>(edm::InputTag("generator"));

  m_observables = parameters_.getParameter<vstring>("observables");
  m_eventObservables = parameters_.getParameter<vstring>("eventObservables");
  m_pfInJetObservables = parameters_.getParameter<vstring>("pfInJetObservables");
  m_npvBins = parameters_.getParameter<vDouble>("NPVBins");

  // List of cuts applied to PFCs that we want to plot
  m_cutList = parameters_.getParameter<vstring>("cutList");
  m_cutList2D = parameters_.getParameter<vstring>("binList2D");
  // List of jet cuts that we apply for the case of plotting PFCs in jets
  m_jetCutList = parameters_.getParameter<vstring>("jetCutList");

  m_eventSelectionMap["dijet"] = &passesDijetSelection;
  m_eventSelectionMap["nocut"] = &passesNoCutSelection;
  m_eventSelectionMap["anomalous"] = &passesAnomalousSelection;

  // Link observable strings to the static functions defined in the header file
  // Many of these are quite trivial, but this enables a simple way to include a
  // variety of observables on-the-fly.
  m_funcMap["pt"] = &getPt;
  m_funcMap["energy"] = getEnergy;
  m_funcMap["eta"] = getEta;
  m_funcMap["abseta"] = getAbsEta;
  m_funcMap["phi"] = getPhi;
  m_funcMap["puppi"] = getPuppiWeight;

  m_funcMap["HCalE_depth1"] = getHcalEnergy_depth1;
  m_funcMap["HCalE_depth2"] = getHcalEnergy_depth2;
  m_funcMap["HCalE_depth3"] = getHcalEnergy_depth3;
  m_funcMap["HCalE_depth4"] = getHcalEnergy_depth4;
  m_funcMap["HCalE_depth5"] = getHcalEnergy_depth5;
  m_funcMap["HCalE_depth6"] = getHcalEnergy_depth6;
  m_funcMap["HCalE_depth7"] = getHcalEnergy_depth7;

  m_funcMap["PS1_E"] = getPS1Energy;
  m_funcMap["PS2_E"] = getPS2Energy;
  m_funcMap["PS_E"] = getPSEnergy;

  m_funcMap["ECal_E"] = getEcalEnergy;
  m_funcMap["RawECal_E"] = getRawEcalEnergy;
  m_funcMap["HCal_E"] = getHcalEnergy;
  m_funcMap["RawHCal_E"] = getRawHcalEnergy;
  m_funcMap["HO_E"] = getHOEnergy;
  m_funcMap["RawHO_E"] = getRawHOEnergy;
  m_funcMap["PFHad_calibration"] = getHadCalibration;

  m_funcMap["MVAIsolated"] = getMVAIsolated;
  m_funcMap["MVAEPi"] = getMVAEPi;
  m_funcMap["MVAEMu"] = getMVAEMu;
  m_funcMap["MVAPiMu"] = getMVAPiMu;
  m_funcMap["MVANothingGamma"] = getMVANothingGamma;
  m_funcMap["MVANothingNH"] = getMVANothingNH;
  m_funcMap["MVAGammaNH"] = getMVAGammaNH;

  m_funcMap["DNNESigIsolated"] = getDNNESigIsolated;
  m_funcMap["DNNESigNonIsolated"] = getDNNESigNonIsolated;
  m_funcMap["DNNEBkgNonIsolated"] = getDNNEBkgNonIsolated;
  m_funcMap["DNNEBkgTauIsolated"] = getDNNEBkgTauIsolated;
  m_funcMap["DNNEBkgPhotonIsolated"] = getDNNEBkgPhotonIsolated;

  m_funcMap["hcalE"] = getHCalEnergy;
  m_funcMap["eOverP"] = getEoverP;
  m_funcMap["nTrkInBlock"] = getNTracksInBlock;

  m_funcMap["trk_nStripHits"] = getTrackNStripHits;
  m_funcMap["trk_nPixHits"] = getTrackNPixHits;
  m_funcMap["trk_chi2"] = getTrackChi2;
  m_funcMap["trk_ptError"] = getTrackPtError;
  m_funcMap["trk_relPtError"] = getTrackRelPtError;
  m_funcMap["trk_d0"] = getTrackD0;
  m_funcMap["trk_z0"] = getTrackZ0;
  m_funcMap["trk_thetaError"] = getTrackThetaError;
  m_funcMap["trk_etaError"] = getTrackEtaError;
  m_funcMap["trk_phiError"] = getTrackPhiError;

  m_eventFuncMap["NPFC"] = getNPFC;
  m_eventFuncMap["MaxPtFrac"] = getMaxPt;
  m_jetWideFuncMap["NPFC"] = getNPFCinJet;
  m_jetWideFuncMap["MaxPtFrac"] = getMaxPtFracJet;

  m_pfInJetFuncMap["PFSpectrum"] = getEnergySpectrum;

  // Link jet observables to static functions in the header file.
  // This is very similar to m_funcMap, but for jets instead.
  m_jetFuncMap["pt"] = getJetPt;
  m_jetFuncMap["chargeFrac"] = getJetChargeFrac;

  m_particleTypeName[reco::PFCandidate::ParticleType::h] = "chargedHadPFC";
  m_particleTypeName[reco::PFCandidate::ParticleType::h0] = "neutralHadPFC";
  m_particleTypeName[reco::PFCandidate::ParticleType::e] = "electronPFC";
  m_particleTypeName[reco::PFCandidate::ParticleType::mu] = "muonPFC";
  m_particleTypeName[reco::PFCandidate::ParticleType::gamma] = "gammaPFC";
  m_particleTypeName[reco::PFCandidate::ParticleType::h_HF] = "hadHFPFC";
  m_particleTypeName[reco::PFCandidate::ParticleType::egamma_HF] = "emHFPFC";

  // Convert the cutList strings into real cuts that can be applied
  // The format should be three comma separated values
  // with the first number being the name of the observable
  // (corresponding to a key in m_funcMap),
  // the second being the minimum value, and the last being the max.
  //
  for (unsigned int i = 0; i < m_cutList.size(); i++) {
    m_fullCutList.push_back(std::vector<std::string>());
    while (m_cutList[i].find(']') != std::string::npos) {
      size_t pos = m_cutList[i].find(']');
      m_fullCutList[i].push_back(m_cutList[i].substr(1, pos));
      m_cutList[i].erase(0, pos + 1);
    }
  }

  for (unsigned int i = 0; i < m_fullCutList.size(); i++) {
    m_binList.push_back(std::vector<std::vector<double>>());
    for (unsigned int j = 0; j < m_fullCutList[i].size(); j++) {
      size_t pos = m_fullCutList[i][j].find(';');
      std::string observableName = m_fullCutList[i][j].substr(0, pos);
      m_fullCutList[i][j].erase(0, pos + 1);

      m_binList[i].push_back(getBinList(m_fullCutList[i][j]));
      m_fullCutList[i][j] = observableName;
    }
  }

  for (unsigned int i = 0; i < m_cutList2D.size(); i++) {
    m_fullCutList2D.push_back(std::vector<std::string>());
    while (m_cutList2D[i].find(']') != std::string::npos) {
      size_t pos = m_cutList2D[i].find(']');
      m_fullCutList2D[i].push_back(m_cutList2D[i].substr(1, pos));
      m_cutList2D[i].erase(0, pos + 1);
    }
  }

  for (unsigned int i = 0; i < m_fullCutList2D.size(); i++) {
    m_binList2D.push_back(std::vector<std::vector<double>>());
    for (unsigned int j = 0; j < m_fullCutList2D[i].size(); j++) {
      size_t pos = m_fullCutList2D[i][j].find(';');
      std::string observableName = m_fullCutList2D[i][j].substr(0, pos);
      m_fullCutList2D[i][j].erase(0, pos + 1);

      m_binList2D[i].push_back(getBinList(m_fullCutList2D[i][j]));
      m_fullCutList2D[i][j] = observableName;
    }
  }

  // Convert the jetCutList strings into real cuts that can be applied
  // The format should be three comma separated values,
  // with the first number being the name of the observable
  // (corresponding to a key in m_jetFuncMap),
  // the second being the minimum value, and the last being the max value.
  //
  for (unsigned int i = 0; i < m_jetCutList.size(); i++) {
    m_fullJetCutList.push_back(std::vector<std::string>());
    while (m_jetCutList[i].find(']') != std::string::npos) {
      size_t pos = m_jetCutList[i].find(']');
      m_fullJetCutList[i].push_back(m_jetCutList[i].substr(1, pos));
      m_jetCutList[i].erase(0, pos + 1);
    }
  }

  for (unsigned int i = 0; i < m_fullJetCutList.size(); i++) {
    m_jetBinList.push_back(std::vector<std::vector<double>>());
    for (unsigned int j = 0; j < m_fullJetCutList[i].size(); j++) {
      size_t pos = m_fullJetCutList[i][j].find(';');
      std::string observableName = m_fullJetCutList[i][j].substr(0, pos);
      m_fullJetCutList[i][j].erase(0, pos + 1);

      m_jetBinList[i].push_back(getBinList(m_fullJetCutList[i][j]));
      m_fullJetCutList[i][j] = observableName;
    }
  }
}

// ***********************************************************
PFAnalyzer::~PFAnalyzer() { LogTrace("PFAnalyzer") << "[PFAnalyzer] Saving the histos"; }

// ***********************************************************
void PFAnalyzer::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const&) {
  ibooker.setCurrentFolder(m_directory);

  for (unsigned int i = 0; i < m_fullCutList.size(); i++) {
    m_allSuffixes.push_back(getAllSuffixes(m_fullCutList[i], m_binList[i]));
  }

  for (unsigned int i = 0; i < m_fullJetCutList.size(); i++) {
    m_allJetSuffixes.push_back(getAllSuffixes(m_fullJetCutList[i], m_jetBinList[i]));
  }

  // Books a 2D histogram for each histogram in the config file.
  // The format for the observables should be four comma separated values,
  // with the first being the observable name (corresponding to one of
  // the keys in m_funcMap), the second being the number of bins,
  // and the last two being the min and max value for the histogram respectively.

  for (unsigned int i = 0; i < m_fullCutList2D.size(); i++) {
    // Loop over all of the different types of PF candidates
    for (unsigned int m = 0; m < m_pfNames.size(); m++) {
      // For each observable, we make a couple histograms based on a few generic categorizations.
      // In all cases, the PFCs that go into these histograms must pass the PFC selection from m_cutList.
      std::string histName =
          Form("%s_%s_%s", m_pfNames[m].c_str(), m_fullCutList2D[i][0].c_str(), m_fullCutList2D[i][1].c_str());
      MonitorElement* mHist =
          ibooker.book2D(histName,
                         Form(";%s;%s", m_fullCutList2D[i][0].c_str(), m_fullCutList2D[i][1].c_str()),
                         m_binList2D[i][0].size(),
                         m_binList2D[i][0][0],
                         m_binList2D[i][0][m_binList2D[i][0].size() - 1],
                         m_binList2D[i][1].size(),
                         m_binList2D[i][1][0],
                         m_binList2D[i][1][m_binList2D[i][0].size() - 1]);
      map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHist));
    }
  }

  for (unsigned int npv = 0; npv < m_npvBins.size() - 1; npv++) {
    std::string npvString = Form("npv_%.0f_%.0f", m_npvBins[npv], m_npvBins[npv + 1]);
    // TODO: Make it possible to use an arbitrary list of bins instead of evenly space bins?
    // It is not clear if this is straightforward to do with these classes and CMSSW.
    // If it is, it should be an easy change to the code.
    //
    //
    // Books a histogram for each histogram in the config file.
    // The format for the observables should be four comma separated values,
    // with the first being the observable name (corresponding to one of
    // the keys in m_funcMap), the second being the number of bins,
    // and the last two being the min and max value for the histogram respectively.
    for (unsigned int i = 0; i < m_observables.size(); i++) {
      std::string cObservable = m_observables[i];
      PFAnalyzer::binInfo obsInfo = getBinInfo(cObservable);

      if (npv == 0)
        m_observableNames.push_back(obsInfo.observable);

      for (unsigned int j = 0; j < m_allSuffixes.size(); j++) {
        for (unsigned int n = 0; n < m_allSuffixes[j].size(); n++) {
          // Loop over all of the different types of PF candidates
          for (unsigned int m = 0; m < m_pfNames.size(); m++) {
            // For each observable, we make a couple histograms based on a few generic categorizations.
            // In all cases, the PFCs that go into these histograms must pass the PFC selection from m_cutList.
            std::string histName = Form("%s_%s%s_%s",
                                        m_pfNames[m].c_str(),
                                        obsInfo.observable.c_str(),
                                        m_allSuffixes[j][n].c_str(),
                                        npvString.c_str());
            MonitorElement* mHist = ibooker.book1D(
                histName, Form(";%s;", obsInfo.axisName.c_str()), obsInfo.nBins, obsInfo.binMin, obsInfo.binMax);
            map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHist));
          }

          for (unsigned int k = 0; k < m_allJetSuffixes.size(); k++) {
            for (unsigned int p = 0; p < m_allJetSuffixes[k].size(); p++) {
              for (unsigned int m = 0; m < m_pfNames.size(); m++) {
                // These histograms are for PFCs passing the basic selection, and which are matched to jets
                // that pass the jet selection
                std::string histName = Form("%s_jetMatched_%s%s_jetCuts%s_%s",
                                            m_pfNames[m].c_str(),
                                            obsInfo.observable.c_str(),
                                            m_allSuffixes[j][n].c_str(),
                                            m_allJetSuffixes[k][p].c_str(),
                                            npvString.c_str());
                MonitorElement* mHistInJet = ibooker.book1D(
                    histName, Form(";%s;", obsInfo.axisName.c_str()), obsInfo.nBins, obsInfo.binMin, obsInfo.binMax);
                map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHistInJet));
              }
            }
          }
        }
      }
    }

    // Do the same for global observables (things like the number of PFCs in an event, or in a jet, etc)
    for (unsigned int i = 0; i < m_eventObservables.size(); i++) {
      std::string cEventObservable = m_eventObservables[i];
      size_t pos = cEventObservable.find(';');
      std::string observableName = cEventObservable.substr(0, pos);
      cEventObservable.erase(0, pos + 1);

      pos = cEventObservable.find(';');
      std::string axisString = cEventObservable.substr(0, pos);
      cEventObservable.erase(0, pos + 1);

      pos = cEventObservable.find(';');
      int nBins = atoi(cEventObservable.substr(0, pos).c_str());
      cEventObservable.erase(0, pos + 1);

      pos = cEventObservable.find(';');
      float binMin = atof(cEventObservable.substr(0, pos).c_str());
      cEventObservable.erase(0, pos + 1);

      pos = cEventObservable.find(';');
      float binMax = atof(cEventObservable.substr(0, pos).c_str());
      cEventObservable.erase(0, pos + 1);

      pos = cEventObservable.find(';');
      int nBinsJet = atoi(cEventObservable.substr(0, pos).c_str());
      cEventObservable.erase(0, pos + 1);

      pos = cEventObservable.find(';');
      float binMinJet = atof(cEventObservable.substr(0, pos).c_str());
      cEventObservable.erase(0, pos + 1);

      float binMaxJet = atof(cEventObservable.c_str());
      if (npv == 0)
        m_eventObservableNames.push_back(observableName);

      for (unsigned int m = 0; m < m_pfNames.size(); m++) {
        std::string histName = Form("%s_%s_%s", m_pfNames[m].c_str(), observableName.c_str(), npvString.c_str());
        MonitorElement* mHist = ibooker.book1D(histName, Form(";%s;", axisString.c_str()), nBins, binMin, binMax);
        map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHist));
      }

      for (unsigned int k = 0; k < m_allJetSuffixes.size(); k++) {
        for (unsigned int p = 0; p < m_allJetSuffixes[k].size(); p++) {
          for (unsigned int m = 0; m < m_pfNames.size(); m++) {
            // These histograms are for PFCs passing the basic selection, and which are matched to jets
            // that pass the jet selection
            std::string histName = Form("%s_jetMatched_%s_jetCuts%s_%s",
                                        m_pfNames[m].c_str(),
                                        observableName.c_str(),
                                        m_allJetSuffixes[k][p].c_str(),
                                        npvString.c_str());
            MonitorElement* mHistInJet =
                ibooker.book1D(histName, Form(";%s;", axisString.c_str()), nBinsJet, binMinJet, binMaxJet);
            map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHistInJet));
          }
        }
      }
    }

    for (unsigned int i = 0; i < m_pfInJetObservables.size(); i++) {
      std::string cPfInJetObservable = m_pfInJetObservables[i];
      PFAnalyzer::binInfo pfInJetInfo = getBinInfo(cPfInJetObservable);
      if (npv == 0)
        m_pfInJetObservableNames.push_back(pfInJetInfo.observable);

      for (unsigned int j = 0; j < m_allSuffixes.size(); j++) {
        for (unsigned int n = 0; n < m_allSuffixes[j].size(); n++) {
          for (unsigned int k = 0; k < m_allJetSuffixes.size(); k++) {
            for (unsigned int p = 0; p < m_allJetSuffixes[k].size(); p++) {
              for (unsigned int m = 0; m < m_pfNames.size(); m++) {
                // These histograms are for PFCs passing the basic selection, and which are matched to jets
                // that pass the jet selection
                std::string histName = Form("%s_jetMatched_%s%s_jetCuts%s_%s",
                                            m_pfNames[m].c_str(),
                                            pfInJetInfo.observable.c_str(),
                                            m_allSuffixes[j][n].c_str(),
                                            m_allJetSuffixes[k][p].c_str(),
                                            npvString.c_str());
                MonitorElement* mHistInJet = ibooker.book1D(histName,
                                                            Form(";%s;", pfInJetInfo.axisName.c_str()),
                                                            pfInJetInfo.nBins,
                                                            pfInJetInfo.binMin,
                                                            pfInJetInfo.binMax);
                map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHistInJet));
              }
            }
          }
        }
      }
    }

    // Extra histograms for basic validation of the selection etc.
    std::string histName = Form("jetPt_%s", npvString.c_str());
    MonitorElement* mHist = ibooker.book1D(histName, Form(";%s;", "p_{T,jet}"), 2000, 0, 2000);
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHist));

    histName = Form("jetPtLead_%s", npvString.c_str());
    mHist = ibooker.book1D(histName, Form(";%s;", "p_{T, leading jet}"), 2000, 0, 2000);
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHist));

    histName = Form("jetEta_%s", npvString.c_str());
    mHist = ibooker.book1D(histName, Form(";%s;", "#eta_{jet}"), 200, -5, 5);
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHist));

    histName = Form("jetEtaLead_%s", npvString.c_str());
    mHist = ibooker.book1D(histName, Form(";%s;", "#eta_{leading jet}"), 200, -5, 5);
    map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHist));
  }

  std::string histName = Form("NPV");
  MonitorElement* mHist = ibooker.book1D(histName, Form(";%s;", "N_PV"), 100, 0, 100);
  map_of_MEs.insert(std::pair<std::string, MonitorElement*>(m_directory + "/" + histName, mHist));
}

PFAnalyzer::binInfo PFAnalyzer::getBinInfo(std::string observableString) {
  PFAnalyzer::binInfo binningDetails;

  size_t pos = observableString.find(';');
  binningDetails.observable = observableString.substr(0, pos);
  observableString.erase(0, pos + 1);

  std::vector<double> binList = getBinList(observableString);
  pos = observableString.find(';');
  binningDetails.axisName = observableString.substr(0, pos);
  observableString.erase(0, pos + 1);

  pos = observableString.find(';');
  binningDetails.nBins = atoi(observableString.substr(0, pos).c_str());
  observableString.erase(0, pos + 1);

  pos = observableString.find(';');
  binningDetails.binMin = atof(observableString.substr(0, pos).c_str());
  observableString.erase(0, pos + 1);

  binningDetails.binMax = atof(observableString.c_str());

  return binningDetails;
}

void PFAnalyzer::bookMESetSelection(std::string DirName, DQMStore::IBooker& ibooker) {
  ibooker.setCurrentFolder(DirName);
}

// ***********************************************************
void PFAnalyzer::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {}

// How many significant digits do we need to save for the values to be distinct?
std::string PFAnalyzer::stringWithDecimals(int bin, std::vector<double> bins) {
  double diff = bins[bin + 1] - bins[bin];
  double sigFigs = log10(diff);

  // We only want to save as many significant digits as we need to.
  // Currently, we might lose some information, so we should think about
  // if we want to specify more digits
  if (sigFigs >= 1) {
    return Form("%.0f_%.0f", bins[bin], bins[bin + 1]);
  }

  int nDecimals = int(-1 * sigFigs) + 1;
  // We do not want to use decimals since these can mess up histogram retrieval in some cases.
  // Instead, we use a 'p' to indicate the decimal.
  double newDigit = std::abs((bins[bin] - int(bins[bin])) * pow(10, nDecimals));
  double newDigit2 = std::abs((bins[bin + 1] - int(bins[bin + 1])) * pow(10, nDecimals));
  std::string signStringLow = "";
  std::string signStringHigh = "";
  if (bins[bin] < 0)
    signStringLow = "m";
  if (bins[bin + 1] < 0)
    signStringHigh = "m";

  int higherDigitsLow = (bins[bin] > 0) ? floor(bins[bin]) : ceil(bins[bin]);
  int higherDigitsHigh = (bins[bin + 1] > 0) ? floor(bins[bin + 1]) : ceil(bins[bin + 1]);

  return Form("%s%dp%.0f_%s%dp%.0f",
              signStringLow.c_str(),
              std::abs(higherDigitsLow),
              newDigit,
              signStringHigh.c_str(),
              std::abs(higherDigitsHigh),
              newDigit2);
}

std::vector<double> PFAnalyzer::getBinList(std::string binString) {
  std::vector<double> binList;

  while (binString.find(';') != std::string::npos) {
    size_t pos = binString.find(';');
    binList.push_back(atof(binString.substr(0, pos).c_str()));
    binString.erase(0, pos + 1);
  }
  binList.push_back(atof(binString.c_str()));

  if (binList.size() == 3) {
    int nBins = int(binList[0]);
    double minVal = binList[1];
    double maxVal = binList[2];
    binList.clear();

    for (int i = 0; i <= nBins; i++) {
      binList.push_back(minVal + i * (maxVal - minVal) / nBins);
    }
  }

  return binList;
}

std::vector<std::string> PFAnalyzer::getAllSuffixes(std::vector<std::string> observables,
                                                    std::vector<std::vector<double>> binnings) {
  int nTotalBins = 1;
  std::vector<int> nBins;
  for (unsigned int i = 0; i < binnings.size(); i++) {
    nTotalBins = (binnings[i].size() - 1) * nTotalBins;
    nBins.push_back(binnings[i].size() - 1);
  }

  std::vector<std::vector<int>> binList;

  binList.reserve(nTotalBins);
  for (int i = 0; i < nTotalBins; i++) {
    binList.push_back(std::vector<int>());
  }

  int factor = nTotalBins;
  int otherFactor = 1;
  for (unsigned int i = 0; i < binnings.size(); i++) {
    factor = factor / nBins[i];

    for (int k = 0; k < factor; k++) {
      for (int j = 0; j < nBins[i]; j++) {
        for (int m = 0; m < otherFactor; m++) {
          int binNumber = k * nBins[i] + j * otherFactor + m;
          binList[binNumber].push_back(j);
        }
      }
    }
    otherFactor = otherFactor * nBins[i];
  }

  std::vector<std::string> allSuffixes;
  allSuffixes.reserve(nTotalBins);
  for (int i = 0; i < nTotalBins; i++) {
    allSuffixes.push_back(getSuffix(binList[i], observables, binnings));
  }

  return allSuffixes;
}

// Get a unique string corresponding to the selection cuts
std::string PFAnalyzer::getSuffix(std::vector<int> binList,
                                  std::vector<std::string> observables,
                                  std::vector<std::vector<double>> binnings) {
  std::string suffix = "";
  for (unsigned int i = 0; i < binList.size(); i++) {
    if (binList[i] < 0)
      return "";
    std::string digitString = stringWithDecimals(binList[i], binnings[i]);

    suffix = Form("%s_%s_%s", suffix.c_str(), observables[i].c_str(), digitString.c_str());
  }

  return suffix;
}

int PFAnalyzer::getBinNumber(double binVal, std::vector<double> bins) {
  if (binVal < bins[0])
    return -1;
  for (unsigned int i = 0; i < bins.size(); i++) {
    if (binVal < bins[i])
      return i - 1;
  }

  return -1;
}

int PFAnalyzer::getBinNumbers(std::vector<double> binVal, std::vector<std::vector<double>> bins) {
  std::vector<int> cbins;
  std::vector<int> nBins;
  for (unsigned int i = 0; i < binVal.size(); i++) {
    int cbin = getBinNumber(binVal[i], bins[i]);
    if (cbin < 0)
      return -1;
    nBins.push_back(bins[i].size() - 1);
    cbins.push_back(cbin);
  }

  int bin = 0;
  int factor = 1;
  for (unsigned int i = 0; i < binVal.size(); i++) {
    bin += cbins[i] * factor;
    factor = factor * nBins[i];
  }

  return bin;
}

int PFAnalyzer::getPFBin(const reco::PFCandidate pfCand,
                         const pat::PackedCandidate packedCand,
                         const reco::CandidatePtr cand,
                         int partType,
                         int i) {
  std::vector<double> binVals;
  binVals.reserve(m_fullCutList[i].size());
  for (unsigned int j = 0; j < m_fullCutList[i].size(); j++) {
    binVals.push_back(m_funcMap[m_fullCutList[i][j]](pfCand, packedCand, cand, partType));
  }

  return getBinNumbers(binVals, m_binList[i]);
}

int PFAnalyzer::getJetBin(const reco::Jet jetCand, const std::vector<reco::PFCandidatePtr> pfCands, int i) {
  std::vector<double> binVals;
  binVals.reserve(m_fullJetCutList[i].size());
  for (unsigned int j = 0; j < m_fullJetCutList[i].size(); j++) {
    binVals.push_back(m_jetFuncMap[m_fullJetCutList[i][j]](jetCand, pfCands));
  }

  return getBinNumbers(binVals, m_jetBinList[i]);
}

// ***********************************************************
void PFAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const edm::Handle<GenEventInfoProduct> genEventInfo = iEvent.getHandle(tok_ew_);
  double eventWeight = 1;
  if (genEventInfo.isValid()) {
    eventWeight = genEventInfo->weight();
  }

  //Vertex information
  edm::Handle<reco::VertexCollection> vertexHandle;
  iEvent.getByToken(vertexToken_, vertexHandle);

  if (!vertexHandle.isValid()) {
    LogDebug("") << "PFAnalyzer: Could not find vertex collection" << std::endl;
  }
  int numPV = 0;

  if (vertexHandle.isValid()) {
    reco::VertexCollection vertex = *(vertexHandle.product());
    for (reco::VertexCollection::const_iterator v = vertex.begin(); v != vertex.end(); ++v) {
      if (v->isFake())
        continue;
      if (v->ndof() < 4)
        continue;
      if (fabs(v->z()) > 24.0)
        continue;
      ++numPV;
    }
  }

  int npvBin = getBinNumber(numPV, m_npvBins);
  if (npvBin < 0)
    return;
  std::string npvString = Form("npv_%.0f_%.0f", m_npvBins[npvBin], m_npvBins[npvBin + 1]);

  // **** Get the TriggerResults container
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken_, triggerResults);
  if (!triggerResults.isValid()) {
    edm::LogError("PFAnalyzer") << "invalid trigger result \n";
    return;
  }
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);

  edm::Handle<reco::PFCandidateCollection> recoPfCollection;
  edm::Handle<pat::PackedCandidateCollection> patPfCollection;
  std::vector<reco::PFCandidate> pfCollection;

  edm::Handle<pat::JetCollection> patJets;
  edm::Handle<reco::PFJetCollection> pfJets;
  std::vector<reco::Jet> jets;

  unsigned int numJets = 0;
  unsigned int numPFCands = 0;
  m_isMiniAOD = true;
  if (!m_isMiniAOD) {
    iEvent.getByToken(thePfCandidateCollection_, recoPfCollection);
    if (!recoPfCollection.isValid()) {
      edm::LogError("PFAnalyzer") << "invalid collection: PF candidate \n";
      return;
    }
    for (unsigned int i = 0; i < recoPfCollection->size(); i++) {
      pfCollection.push_back(recoPfCollection->at(i));
    }
    numPFCands = recoPfCollection->size();

    iEvent.getByToken(pfJetsToken_, pfJets);
    if (!pfJets.isValid()) {
      edm::LogError("PFAnalyzer") << "invalid collection: PF jets \n";
      return;
    }
    numJets = pfJets->size();
    for (unsigned int i = 0; i < numJets; i++) {
      jets.push_back(pfJets->at(i));
    }
  } else {
    iEvent.getByToken(patJetsToken_, patJets);
    if (!patJets.isValid()) {
      edm::LogError("PFAnalyzer") << "invalid collection: PF jets \n";
      return;
    }
    numJets = patJets->size();
    for (unsigned int i = 0; i < numJets; i++) {
      jets.push_back(patJets->at(i));
    }

    iEvent.getByToken(patPfCandidateCollection_, patPfCollection);
    if (!patPfCollection.isValid()) {
      edm::LogError("PFAnalyzer") << "invalid collection: PF candidate \n";
      return;
    }
    numPFCands = patPfCollection->size();
  }

  //iEvent.getByToken(puppiWeightToken_, puppiWeight);
  //if(!puppiWeight.isValid()){
  //  edm::LogError("PFAnalyzer") << "invalid collection: Puppi weights \n";
  //}

  if (!passesTriggerSelection(jets, triggerResults, triggerNames, m_triggerOptions)) {
    return;
  }

  if (!m_eventSelectionMap[m_selection](jets)) {
    return;
  }

  for (unsigned int i_pfcand = 0; i_pfcand < numPFCands; i_pfcand++) {
    reco::PFCandidate recoPF;
    pat::PackedCandidate packedCand;
    reco::CandidatePtr cand;
    int partType = 0;
    if (m_isMiniAOD) {
      packedCand = patPfCollection->at(i_pfcand);
      partType = 1;
    } else {
      recoPF = pfCollection[i_pfcand];
    }

    for (unsigned int j = 0; j < m_fullCutList.size(); j++) {
      int binNumber = getPFBin(recoPF, packedCand, cand, partType, j);

      if (binNumber < 0)
        continue;
      if (binNumber >= int(m_allSuffixes[j].size())) {
        continue;
      }
      std::string binString = m_allSuffixes[j][binNumber];

      for (unsigned int i = 0; i < m_fullCutList2D.size(); i++) {
        // For each observable, we make a couple histograms based on a few generic categorizations.
        // In all cases, the PFCs that go into these histograms must pass the PFC selection from m_cutList.
        std::string histName = Form("%s_%s", m_fullCutList2D[i][0].c_str(), m_fullCutList2D[i][1].c_str());
        double valX = m_funcMap[m_fullCutList2D[i][0]](recoPF, packedCand, cand, partType);
        double valY = m_funcMap[m_fullCutList2D[i][1]](recoPF, packedCand, cand, partType);

        map_of_MEs[m_directory + "/allPFC_" + histName]->Fill(valX, valY, eventWeight);
        if (partType == 0 && m_particleTypeName.find(recoPF.particleId()) != m_particleTypeName.end()) {
          map_of_MEs[m_directory + "/" + m_particleTypeName[recoPF.particleId()] + "_" + histName]->Fill(
              valX, valY, eventWeight);
        }
      }

      // Eventually, we might want the hist name to include the cuts that we are applying,
      // so I am keepking it as a separate string for now, even though it is redundant.
      // Make plots of all observables
      for (unsigned int i = 0; i < m_observables.size(); i++) {
        std::string histName = Form("%s%s_%s", m_observableNames[i].c_str(), binString.c_str(), npvString.c_str());
        double val = m_funcMap[m_observableNames[i]](recoPF, packedCand, cand, partType);
        map_of_MEs[m_directory + "/allPFC_" + histName]->Fill(val, eventWeight);

        if (partType == 0 && m_particleTypeName.find(recoPF.particleId()) != m_particleTypeName.end()) {
          map_of_MEs[m_directory + "/" + m_particleTypeName[recoPF.particleId()] + "_" + histName]->Fill(val,
                                                                                                         eventWeight);
        }
      }
    }
  }

  for (unsigned int i = 0; i < m_eventObservableNames.size(); i++) {
    std::string histName = Form("%s_%s", m_eventObservableNames[i].c_str(), npvString.c_str());
    map_of_MEs[m_directory + "/allPFC_" + histName]->Fill(
        m_eventFuncMap[m_eventObservableNames[i]](pfCollection, reco::PFCandidate::ParticleType::X), eventWeight);

    for (const auto& mypair : m_particleTypeName) {
      map_of_MEs[m_directory + "/" + mypair.second + "_" + histName]->Fill(
          m_eventFuncMap[m_eventObservableNames[i]](pfCollection, mypair.first), eventWeight);
    }
  }

  // Plots for generic debugging
  map_of_MEs[m_directory + "/NPV"]->Fill(numPV, eventWeight);
  reco::Jet leadJet;
  if (m_isMiniAOD) {
    leadJet = *patJets->begin();
  } else {
    leadJet = *pfJets->begin();
  }
  map_of_MEs[m_directory + Form("/jetPtLead_%s", npvString.c_str())]->Fill(leadJet.pt(), eventWeight);
  map_of_MEs[m_directory + Form("/jetEtaLead_%s", npvString.c_str())]->Fill(leadJet.eta(), eventWeight);

  // Make plots of all observables, this time for PF candidates within jets
  for (unsigned int index = 0; index < numJets; index++) {
    reco::Jet cjet;
    if (m_isMiniAOD) {
      cjet = patJets->at(index);
    } else {
      cjet = pfJets->at(index);
    }
    std::vector<reco::PFCandidatePtr> pfConstits;
    std::vector<reco::CandidatePtr> patConstits;
    unsigned int nConstit = 0;
    if (m_isMiniAOD) {
      patConstits = patJets->at(index).daughterPtrVector();
      nConstit = patConstits.size();
    } else {
      pfConstits = pfJets->at(index).getPFConstituents();
      nConstit = pfConstits.size();
    }

    map_of_MEs[m_directory + Form("/jetPt_%s", npvString.c_str())]->Fill(cjet.pt(), eventWeight);
    map_of_MEs[m_directory + Form("/jetEta_%s", npvString.c_str())]->Fill(cjet.eta(), eventWeight);

    for (unsigned int k = 0; k < m_fullJetCutList.size(); k++) {
      pat::PackedCandidate packedCand;
      reco::CandidatePtr cand;
      reco::PFCandidatePtr recoPFPtr;
      reco::PFCandidate recoPF;
      int jetBinNumber = getJetBin(cjet, pfConstits, k);
      if (jetBinNumber < 0)
        continue;
      std::string jetBinString = m_allJetSuffixes[k][jetBinNumber];

      for (unsigned int iConstit = 0; iConstit < nConstit; iConstit++) {
        int partType = 0;
        if (m_isMiniAOD) {
          cand = patConstits[iConstit];
          partType = 2;
        } else {
          recoPFPtr = pfConstits[iConstit];
          recoPF = *recoPFPtr;
        }
        for (unsigned int j = 0; j < m_fullCutList.size(); j++) {
          int binNumber = getPFBin(recoPF, packedCand, cand, partType, j);
          if (binNumber < 0)
            continue;
          if (binNumber >= int(m_allSuffixes[j].size())) {
            continue;
          }
          std::string binString = m_allSuffixes[j][binNumber];

          for (unsigned int i = 0; i < m_observableNames.size(); i++) {
            std::string histName = Form("%s%s_jetCuts%s_%s",
                                        m_observableNames[i].c_str(),
                                        binString.c_str(),
                                        jetBinString.c_str(),
                                        npvString.c_str());
            map_of_MEs[m_directory + "/allPFC_jetMatched_" + histName]->Fill(
                m_funcMap[m_observableNames[i]](recoPF, packedCand, cand, partType), eventWeight);
            if (partType == 0 && m_particleTypeName.find(recoPF.particleId()) != m_particleTypeName.end()) {
              map_of_MEs[m_directory + "/" + m_particleTypeName[recoPF.particleId()] + "_jetMatched_" + histName]->Fill(
                  m_funcMap[m_observableNames[i]](recoPF, packedCand, cand, partType), eventWeight);
            }
          }

          for (unsigned int i = 0; i < m_pfInJetObservableNames.size(); i++) {
            std::string histName = Form("%s%s_jetCuts%s_%s",
                                        m_pfInJetObservableNames[i].c_str(),
                                        binString.c_str(),
                                        jetBinString.c_str(),
                                        npvString.c_str());
            map_of_MEs[m_directory + "/allPFC_jetMatched_" + histName]->Fill(
                m_pfInJetFuncMap[m_pfInJetObservableNames[i]](recoPF, cjet), eventWeight);

            if (partType == 0 && m_particleTypeName.find(recoPF.particleId()) != m_particleTypeName.end()) {
              map_of_MEs[m_directory + "/" + m_particleTypeName[recoPF.particleId()] + "_jetMatched_" + histName]->Fill(
                  m_pfInJetFuncMap[m_pfInJetObservableNames[i]](recoPF, cjet), eventWeight);
            }
          }
        }

        for (unsigned int i = 0; i < m_eventObservableNames.size(); i++) {
          std::string histName =
              Form("%s_jetCuts%s_%s", m_eventObservableNames[i].c_str(), jetBinString.c_str(), npvString.c_str());
          map_of_MEs[m_directory + "/allPFC_jetMatched_" + histName]->Fill(
              m_jetWideFuncMap[m_eventObservableNames[i]](pfConstits, reco::PFCandidate::ParticleType::X, cjet),
              eventWeight);
          for (const auto& mypair : m_particleTypeName) {
            map_of_MEs[m_directory + "/" + mypair.second + "_jetMatched_" + histName]->Fill(
                m_jetWideFuncMap[m_eventObservableNames[i]](pfConstits, mypair.first, cjet), eventWeight);
          }
        }
      }
    }
  }
}
