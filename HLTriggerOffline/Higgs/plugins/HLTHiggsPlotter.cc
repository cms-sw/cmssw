
/** \file HLTHiggsPlotter.cc
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "HLTHiggsPlotter.h"
#include "HLTHiggsSubAnalysis.h"
#include "EVTColContainer.h"

#include "TPRegexp.h"

#include <set>
#include <cctype>

HLTHiggsPlotter::HLTHiggsPlotter(const edm::ParameterSet &pset,
                                 const std::string &hltPath,
                                 const std::vector<unsigned int> &objectsType,
                                 const unsigned int &NptPlots,
                                 const std::vector<double> &NminOneCuts)
    : _hltPath(hltPath),
      _hltProcessName(pset.getParameter<std::string>("hltProcessName")),
      _objectsType(std::set<unsigned int>(objectsType.begin(), objectsType.end())),
      _nObjects(objectsType.size()),
      _parametersEta(pset.getParameter<std::vector<double> >("parametersEta")),
      _parametersPhi(pset.getParameter<std::vector<double> >("parametersPhi")),
      _parametersTurnOn(pset.getParameter<std::vector<double> >("parametersTurnOn")),
      _NptPlots(NptPlots),
      _NminOneCuts(NminOneCuts) {
  for (std::set<unsigned int>::iterator it = _objectsType.begin(); it != _objectsType.end(); ++it) {
    // Some parameters extracted from the .py
    std::string objStr = EVTColContainer::getTypeString(*it);
    _cutMinPt[*it] = pset.getParameter<double>(std::string(objStr + "_cutMinPt").c_str());
    _cutMaxEta[*it] = pset.getParameter<double>(std::string(objStr + "_cutMaxEta").c_str());
  }
}

HLTHiggsPlotter::~HLTHiggsPlotter() {}

void HLTHiggsPlotter::beginJob() {}

void HLTHiggsPlotter::beginRun(const edm::Run &iRun, const edm::EventSetup &iSetup) {}

void HLTHiggsPlotter::bookHistograms(DQMStore::IBooker &ibooker, const bool &useNminOneCuts) {
  for (std::set<unsigned int>::iterator it = _objectsType.begin(); it != _objectsType.end(); ++it) {
    std::vector<std::string> sources(2);
    sources[0] = "gen";
    sources[1] = "rec";
    TString maxPt;

    const std::string objTypeStr = EVTColContainer::getTypeString(*it);

    for (size_t i = 0; i < sources.size(); i++) {
      std::string source = sources[i];

      if (useNminOneCuts && *it == EVTColContainer::PFJET) {
        if (source == "gen")
          continue;
        else {
          // N-1 jet plots (dEtaqq, mqq, dPhibb, CSV1, maxCSV_jets, maxCSV_E, PFMET, pt1, pt2, pt3, pt4)
          if (_NminOneCuts[0])
            bookHist(source, objTypeStr, "dEtaqq", ibooker);
          if (_NminOneCuts[1])
            bookHist(source, objTypeStr, "mqq", ibooker);
          if (_NminOneCuts[2])
            bookHist(source, objTypeStr, "dPhibb", ibooker);
          if (_NminOneCuts[3]) {
            if (_NminOneCuts[6])
              bookHist(source, objTypeStr, "maxCSV", ibooker);
            else
              bookHist(source, objTypeStr, "CSV1", ibooker);
          }
          if (_NminOneCuts[4])
            bookHist(source, objTypeStr, "CSV2", ibooker);
          if (_NminOneCuts[5])
            bookHist(source, objTypeStr, "CSV3", ibooker);
        }
      }

      bookHist(source, objTypeStr, "Eta", ibooker);
      bookHist(source, objTypeStr, "Phi", ibooker);
      for (unsigned int i = 0; i < _NptPlots; i++) {
        maxPt = "MaxPt";
        maxPt += i + 1;
        bookHist(source, objTypeStr, maxPt.Data(), ibooker);
      }
    }
  }
}

void HLTHiggsPlotter::analyze(const bool &isPassTrigger,
                              const std::string &source,
                              const std::vector<MatchStruct> &matches,
                              const unsigned int &minCandidates) {
  if (!isPassTrigger) {
    return;
  }
  std::map<unsigned int, int> countobjects;
  // Initializing the count of the used object
  for (std::set<unsigned int>::iterator co = _objectsType.begin(); co != _objectsType.end(); ++co) {
    countobjects[*co] = 0;
  }

  int counttotal = 0;
  const int totalobjectssize2 = _NptPlots * countobjects.size();
  // Fill the histos if pass the trigger (just the two with higher pt)
  for (size_t j = 0; j < matches.size(); ++j) {
    // Is this object owned by this trigger? If not we are not interested...
    if (_objectsType.find(matches[j].objType) == _objectsType.end()) {
      continue;
    }

    const unsigned int objType = matches[j].objType;
    const std::string objTypeStr = EVTColContainer::getTypeString(matches[j].objType);

    float pt = matches[j].pt;
    float eta = matches[j].eta;
    float phi = matches[j].phi;

    TString maxPt;
    if ((unsigned)countobjects[objType] < _NptPlots) {
      maxPt = "MaxPt";
      maxPt += (countobjects[objType] + 1);
      this->fillHist(isPassTrigger, source, objTypeStr, maxPt.Data(), pt);
      // Filled the high pt ...
      ++(countobjects[objType]);
      ++counttotal;
    } else {
      if ((unsigned)countobjects[objType] < minCandidates) {  // To get correct results for HZZ
        ++(countobjects[objType]);
        ++counttotal;
      } else
        continue;  //   Otherwise too many entries in Eta and Phi distributions
    }

    this->fillHist(isPassTrigger, source, objTypeStr, "Eta", eta);
    this->fillHist(isPassTrigger, source, objTypeStr, "Phi", phi);

    if (counttotal == totalobjectssize2) {
      break;
    }
  }
}

void HLTHiggsPlotter::analyze(const bool &isPassTrigger,
                              const std::string &source,
                              const std::vector<MatchStruct> &matches,
                              std::map<std::string, bool> &nMinOne,
                              const float &dEtaqq,
                              const float &mqq,
                              const float &dPhibb,
                              const float &CSV1,
                              const float &CSV2,
                              const float &CSV3,
                              const bool &passAllCuts) {
  if (!isPassTrigger) {
    return;
  }
  std::map<unsigned int, int> countobjects;
  // Initializing the count of the used object
  for (std::set<unsigned int>::iterator co = _objectsType.begin(); co != _objectsType.end(); ++co) {
    if (!(*co == EVTColContainer::PFJET && source == "gen"))  // genJets are not there
      countobjects[*co] = 0;
  }

  int counttotal = 0;
  const int totalobjectssize2 = _NptPlots * countobjects.size();
  // Fill the histos if pass the trigger (just the two with higher pt)
  for (size_t j = 0; j < matches.size(); ++j) {
    // Is this object owned by this trigger? If not we are not interested...
    if (_objectsType.find(matches[j].objType) == _objectsType.end()) {
      continue;
    }

    const unsigned int objType = matches[j].objType;
    const std::string objTypeStr = EVTColContainer::getTypeString(matches[j].objType);

    float pt = matches[j].pt;
    float eta = matches[j].eta;
    float phi = matches[j].phi;

    // PFMET N-1 cut
    if (objType == EVTColContainer::PFMET && _NminOneCuts[8] && !nMinOne["PFMET"])
      continue;

    TString maxPt;
    if ((unsigned)(countobjects)[objType] < _NptPlots) {
      maxPt = "MaxPt";
      maxPt += (countobjects[objType] + 1);
      if (objType != EVTColContainer::PFJET || nMinOne[maxPt.Data()]) {
        this->fillHist(isPassTrigger, source, objTypeStr, maxPt.Data(), pt);
      }
      ++(countobjects[objType]);
      ++counttotal;
    } else
      continue;  // if not needed (minCandidates == _NptPlots if _useNminOneCuts
    if (objType != EVTColContainer::PFJET || passAllCuts) {
      this->fillHist(isPassTrigger, source, objTypeStr, "Eta", eta);
      this->fillHist(isPassTrigger, source, objTypeStr, "Phi", phi);
    }

    if (counttotal == totalobjectssize2) {
      break;
    }
  }
  if (source == "rec" && _objectsType.find(EVTColContainer::PFJET) != _objectsType.end()) {
    if (_NminOneCuts[0] && nMinOne["dEtaqq"]) {
      this->fillHist(isPassTrigger, source, EVTColContainer::getTypeString(EVTColContainer::PFJET), "dEtaqq", dEtaqq);
    }
    if (_NminOneCuts[1] && nMinOne["mqq"]) {
      this->fillHist(isPassTrigger, source, EVTColContainer::getTypeString(EVTColContainer::PFJET), "mqq", mqq);
    }
    if (_NminOneCuts[2] && nMinOne["dPhibb"]) {
      this->fillHist(isPassTrigger, source, EVTColContainer::getTypeString(EVTColContainer::PFJET), "dPhibb", dPhibb);
    }
    if (_NminOneCuts[3]) {
      std::string nameCSVplot = "CSV1";
      if (_NminOneCuts[6])
        nameCSVplot = "maxCSV";
      if (nMinOne[nameCSVplot])
        this->fillHist(
            isPassTrigger, source, EVTColContainer::getTypeString(EVTColContainer::PFJET), nameCSVplot, CSV1);
    }
    if (_NminOneCuts[4] && nMinOne["CSV2"]) {
      this->fillHist(isPassTrigger, source, EVTColContainer::getTypeString(EVTColContainer::PFJET), "CSV2", CSV2);
    }
    if (_NminOneCuts[5] && nMinOne["CSV3"]) {
      this->fillHist(isPassTrigger, source, EVTColContainer::getTypeString(EVTColContainer::PFJET), "CSV3", CSV3);
    }
  }
}

void HLTHiggsPlotter::bookHist(const std::string &source,
                               const std::string &objType,
                               const std::string &variable,
                               DQMStore::IBooker &ibooker) {
  std::string sourceUpper = source;
  sourceUpper[0] = std::toupper(sourceUpper[0]);
  std::string name = source + objType + variable + "_" + _hltPath;
  TH1F *h = nullptr;

  if (variable.find("MaxPt") != std::string::npos) {
    std::string desc;
    if (variable == "MaxPt1")
      desc = "Leading";
    else if (variable == "MaxPt2")
      desc = "Next-to-Leading";
    else
      desc = variable.substr(5, 6) + "th Leading";
    std::string title = "pT of " + desc + " " + sourceUpper + " " + objType +
                        " "
                        "where event pass the " +
                        _hltPath;
    const size_t nBinsStandard = _parametersTurnOn.size() - 1;
    size_t nBins = nBinsStandard;
    float *edges = new float[nBinsStandard + 1];
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
  } else {
    if (variable == "dEtaqq") {
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
      std::string title = symbol + " of " + sourceUpper + " " + objType + " " + "where event pass the " + _hltPath;
      std::vector<double> params = (variable == "Eta") ? _parametersEta : _parametersPhi;

      int nBins = (int)params[0];
      double min = params[1];
      double max = params[2];
      h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
    }
  }
  h->Sumw2();
  _elements[name] = ibooker.book1D(name.c_str(), h);
  delete h;
}

void HLTHiggsPlotter::fillHist(const bool &passTrigger,
                               const std::string &source,
                               const std::string &objType,
                               const std::string &variable,
                               const float &value) {
  std::string sourceUpper = source;
  sourceUpper[0] = toupper(sourceUpper[0]);
  std::string name = source + objType + variable + "_" + _hltPath;

  _elements[name]->Fill(value);
}
