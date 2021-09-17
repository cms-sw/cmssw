/** \file HLTExoticaPlotter.cc
 */

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HLTriggerOffline/Exotica/interface/HLTExoticaPlotter.h"
#include "HLTriggerOffline/Exotica/interface/HLTExoticaSubAnalysis.h"
#include "HLTriggerOffline/Exotica/src/EVTColContainer.cc"

#include "TPRegexp.h"

#include "HLTriggerOffline/Exotica/src/EVTColContainer.cc"

#include <cctype>
#include <set>

HLTExoticaPlotter::HLTExoticaPlotter(const edm::ParameterSet &pset,
                                     const std::string &hltPath,
                                     const std::vector<unsigned int> &objectsType)
    : _hltPath(hltPath),
      _hltProcessName(pset.getParameter<std::string>("hltProcessName")),
      _objectsType(std::set<unsigned int>(objectsType.begin(), objectsType.end())),
      _nObjects(objectsType.size()),
      _parametersEta(pset.getParameter<std::vector<double>>("parametersEta")),
      _parametersPhi(pset.getParameter<std::vector<double>>("parametersPhi")),
      _parametersTurnOn(pset.getParameter<std::vector<double>>("parametersTurnOn")),
      _parametersTurnOnSumEt(pset.getParameter<std::vector<double>>("parametersTurnOnSumEt")),
      _parametersDxy(pset.getParameter<std::vector<double>>("parametersDxy")),
      _drop_pt2(false),
      _drop_pt3(false) {
  if (pset.exists("dropPt2")) {
    _drop_pt2 = pset.getParameter<bool>("dropPt2");
  }
  if (pset.exists("dropPt3")) {
    _drop_pt3 = pset.getParameter<bool>("dropPt3");
  }
  LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::constructor()";
}

HLTExoticaPlotter::~HLTExoticaPlotter() {}

void HLTExoticaPlotter::beginJob() {}

void HLTExoticaPlotter::plotterBookHistos(DQMStore::IBooker &iBooker,
                                          const edm::Run &iRun,
                                          const edm::EventSetup &iSetup) {
  LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::plotterBookHistos()";
  for (std::set<unsigned int>::iterator it = _objectsType.begin(); it != _objectsType.end(); ++it) {
    std::vector<std::string> sources(2);
    sources[0] = "gen";
    sources[1] = "rec";

    const std::string objTypeStr = EVTColContainer::getTypeString(*it);

    for (size_t i = 0; i < sources.size(); i++) {
      std::string source = sources[i];

      if (source == "gen") {
        if (TString(objTypeStr).Contains("MET") || TString(objTypeStr).Contains("MHT") ||
            TString(objTypeStr).Contains("Jet")) {
          continue;
        } else {
          bookHist(iBooker, source, objTypeStr, "MaxPt1");
          if (!_drop_pt2)
            bookHist(iBooker, source, objTypeStr, "MaxPt2");
          if (!_drop_pt3)
            bookHist(iBooker, source, objTypeStr, "MaxPt3");
          bookHist(iBooker, source, objTypeStr, "Eta");
          bookHist(iBooker, source, objTypeStr, "Phi");

          // If the target is electron or muon,
          // we will add Dxy plots.
          if (*it == EVTColContainer::ELEC || *it == EVTColContainer::MUON || *it == EVTColContainer::MUTRK) {
            bookHist(iBooker, source, objTypeStr, "Dxy");
          }
        }
      } else {  // reco
        if (TString(objTypeStr).Contains("MET") || TString(objTypeStr).Contains("MHT")) {
          bookHist(iBooker, source, objTypeStr, "MaxPt1");
          bookHist(iBooker, source, objTypeStr, "SumEt");
        } else {
          bookHist(iBooker, source, objTypeStr, "MaxPt1");
          if (!_drop_pt2)
            bookHist(iBooker, source, objTypeStr, "MaxPt2");
          if (!_drop_pt3)
            bookHist(iBooker, source, objTypeStr, "MaxPt3");
          bookHist(iBooker, source, objTypeStr, "Eta");
          bookHist(iBooker, source, objTypeStr, "Phi");

          // If the target is electron or muon,
          // we will add Dxy plots.
          if (*it == EVTColContainer::ELEC || *it == EVTColContainer::MUON || *it == EVTColContainer::MUTRK) {
            bookHist(iBooker, source, objTypeStr, "Dxy");
          }
        }
      }
    }
  }
}

void HLTExoticaPlotter::analyze(const bool &isPassTrigger,
                                const std::string &source,
                                const std::vector<reco::LeafCandidate> &matches,
                                std::map<int, double> theSumEt,
                                std::vector<float> &dxys) {
  LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::analyze()";
  if (!isPassTrigger) {
    return;
  }

  std::map<unsigned int, int> countobjects;
  // Initializing the count of the used object
  for (std::set<unsigned int>::iterator co = _objectsType.begin(); co != _objectsType.end(); ++co) {
    countobjects[*co] = 0;
  }

  int counttotal = 0;

  // 3 : pt1, pt2, pt3
  int totalobjectssize = 1;
  if (!_drop_pt2)
    totalobjectssize++;
  if (!_drop_pt3)
    totalobjectssize++;
  totalobjectssize *= countobjects.size();
  // Fill the histos if pass the trigger (just the two with higher pt)
  unsigned int jaux = 0;
  // jaux is being used as a dedicated counter to avoid getting
  // a non-existent element inside dxys
  // more information in the issue https://github.com/cms-sw/cmssw/issues/32550
  for (size_t j = 0; j < matches.size(); ++j) {
    // Is this object owned by this trigger? If not we are not interested...
    if (_objectsType.find(matches[j].pdgId()) == _objectsType.end()) {
      ++jaux;
      continue;
    }

    const unsigned int objType = matches[j].pdgId();
    const std::string objTypeStr = EVTColContainer::getTypeString(objType);

    float pt = matches[j].pt();
    float eta = matches[j].eta();
    float phi = matches[j].phi();

    if (!(TString(objTypeStr).Contains("MET") || TString(objTypeStr).Contains("MHT"))) {
      this->fillHist(isPassTrigger, source, objTypeStr, "Eta", eta);
      this->fillHist(isPassTrigger, source, objTypeStr, "Phi", phi);
    } else if (source != "gen") {
      if (theSumEt[objType] >= 0 && countobjects[objType] == 0) {
        this->fillHist(isPassTrigger, source, objTypeStr, "SumEt", theSumEt[objType]);
      }
    }

    if (!dxys.empty() &&
        (objType == EVTColContainer::ELEC || objType == EVTColContainer::MUON || objType == EVTColContainer::MUTRK)) {
      this->fillHist(isPassTrigger, source, objTypeStr, "Dxy", dxys[jaux]);
      ++jaux;
    }

    if (countobjects[objType] == 0) {
      if (!(TString(objTypeStr).Contains("MET") || TString(objTypeStr).Contains("MHT")) || source != "gen") {
        this->fillHist(isPassTrigger, source, objTypeStr, "MaxPt1", pt);
      }
      // Filled the high pt ...
      ++(countobjects[objType]);
      ++counttotal;
    } else if (countobjects[objType] == 1 && !_drop_pt2) {
      if (!(TString(objTypeStr).Contains("MET") || TString(objTypeStr).Contains("MHT"))) {
        this->fillHist(isPassTrigger, source, objTypeStr, "MaxPt2", pt);
      }
      // Filled the second high pt ...
      ++(countobjects[objType]);
      ++counttotal;
    } else if (countobjects[objType] == 2 && !_drop_pt3) {
      if (!(TString(objTypeStr).Contains("MET") || TString(objTypeStr).Contains("MHT"))) {
        this->fillHist(isPassTrigger, source, objTypeStr, "MaxPt3", pt);
      }
      // Filled the third highest pt ...
      ++(countobjects[objType]);
      ++counttotal;
    } else {
      if (counttotal == totalobjectssize) {
        break;
      }
    }

  }  // end loop over matches
}

void HLTExoticaPlotter::bookHist(DQMStore::IBooker &iBooker,
                                 const std::string &source,
                                 const std::string &objType,
                                 const std::string &variable) {
  LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::bookHist()";
  std::string sourceUpper = source;
  sourceUpper[0] = std::toupper(sourceUpper[0]);
  std::string name = source + objType + variable + "_" + _hltPath;
  TH1F *h = nullptr;

  if (variable.find("SumEt") != std::string::npos) {
    std::string title = "Sum ET of " + sourceUpper + " " + objType;
    const size_t nBins = _parametersTurnOnSumEt.size() - 1;
    float *edges = new float[nBins + 1];
    for (size_t i = 0; i < nBins + 1; i++) {
      edges[i] = _parametersTurnOnSumEt[i];
    }
    h = new TH1F(name.c_str(), title.c_str(), nBins, edges);
    delete[] edges;
  } else if (variable.find("Dxy") != std::string::npos) {
    std::string title = "Dxy " + sourceUpper + " " + objType;
    int nBins = _parametersDxy[0];
    double min = _parametersDxy[1];
    double max = _parametersDxy[2];
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  } else if (variable.find("MaxPt") != std::string::npos) {
    std::string desc;  //=
    //        (variable == "MaxPt1") ? "Leading" : (variable == "MaxPt2") ? "Next-to-Leading" : "Next-to-next-to-Leading";
    if (variable == "MaxPt1") {
      desc = "Leading";
    } else if (variable == "MaxPt2") {
      desc = "Next-to-Leading";
    } else {
      desc = "Next-to-next-to-Leading";
    }
    std::string title = "pT of " + desc + " " + sourceUpper + " " + objType +
                        " "
                        "where event pass the " +
                        _hltPath;
    const size_t nBins = _parametersTurnOn.size() - 1;
    float *edges = new float[nBins + 1];
    for (size_t i = 0; i < nBins + 1; i++) {
      edges[i] = _parametersTurnOn[i];
    }
    h = new TH1F(name.c_str(), title.c_str(), nBins, edges);
    delete[] edges;
  }

  else {
    std::string symbol = (variable == "Eta") ? "#eta" : "#phi";
    std::string title = symbol + " of " + sourceUpper + " " + objType + " " + "where event pass the " + _hltPath;
    std::vector<double> params = (variable == "Eta") ? _parametersEta : _parametersPhi;

    int nBins = (int)params[0];
    double min = params[1];
    double max = params[2];
    h = new TH1F(name.c_str(), title.c_str(), nBins, min, max);
  }

  h->Sumw2();
  _elements[name] = iBooker.book1D(name, h);
  //    LogDebug("ExoticaValidation") << "                        booked histo
  //    with name " << name << "\n"
  //				  << "                        at location " <<
  //(unsigned long int)_elements[name];
  delete h;
}

void HLTExoticaPlotter::fillHist(const bool &passTrigger,
                                 const std::string &source,
                                 const std::string &objType,
                                 const std::string &variable,
                                 const float &value) {
  std::string sourceUpper = source;
  sourceUpper[0] = toupper(sourceUpper[0]);
  std::string name = source + objType + variable + "_" + _hltPath;

  LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::fillHist()" << name << " " << value;
  _elements[name]->Fill(value);
  LogDebug("ExoticaValidation") << "In HLTExoticaPlotter::fillHist()" << name << " worked";
}
