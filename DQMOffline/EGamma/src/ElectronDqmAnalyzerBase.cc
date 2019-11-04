
#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"
//#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include <iostream>
#include <algorithm>
#include <sstream>

ElectronDqmAnalyzerBase::ElectronDqmAnalyzerBase(const edm::ParameterSet &conf)
    : bookPrefix_("ele"), bookIndex_(0), histoNamesReady(false) {
  verbosity_ = conf.getUntrackedParameter<int>("Verbosity");
  finalStep_ = conf.getParameter<std::string>("FinalStep");
  inputFile_ = conf.getParameter<std::string>("InputFile");
  outputFile_ = conf.getParameter<std::string>("OutputFile");
  inputInternalPath_ = conf.getParameter<std::string>("InputFolderName");
  outputInternalPath_ = conf.getParameter<std::string>("OutputFolderName");
}

ElectronDqmAnalyzerBase::~ElectronDqmAnalyzerBase() {}

void ElectronDqmAnalyzerBase::setBookPrefix(const std::string &prefix) { bookPrefix_ = prefix; }

void ElectronDqmAnalyzerBase::setBookIndex(short index) { bookIndex_ = index; }

void ElectronDqmAnalyzerBase::setBookEfficiencyFlag(const bool &eff_flag) { bookEfficiencyFlag_ = eff_flag; }

void ElectronDqmAnalyzerBase::setBookStatOverflowFlag(const bool &statOverflow_flag) {
  bookStatOverflowFlag_ = statOverflow_flag;
}

std::string ElectronDqmAnalyzerBase::newName(const std::string &name) {
  if (bookPrefix_.empty()) {
    return name;
  }
  std::ostringstream oss;
  oss << bookPrefix_;
  if (bookIndex_ >= 0) {
    oss << bookIndex_++;
  }
  oss << "_" << name;
  return oss.str();
}

void ElectronDqmAnalyzerBase::bookHistograms(DQMStore::IBooker &ibooker_, edm::Run const &, edm::EventSetup const &) {
  edm::LogInfo("DQMAnalyzeBase::bookHistograms") << std::endl;
}

ElectronDqmAnalyzerBase::MonitorElement *ElectronDqmAnalyzerBase::bookH1(DQMStore::IBooker &iBooker,
                                                                         const std::string &name,
                                                                         const std::string &title,
                                                                         int nchX,
                                                                         double lowX,
                                                                         double highX,
                                                                         const std::string &titleX,
                                                                         const std::string &titleY,
                                                                         Option_t *option) {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement *me = iBooker.book1D(newName(name), title, nchX, lowX, highX);
  if (!titleX.empty()) {
    me->setAxisTitle(titleX);
  }
  if (!titleY.empty()) {
    me->setAxisTitle(titleY, 2);
  }
  if (TString(option) != "") {
    me->setOption(option);
  }
  if (bookStatOverflowFlag_) {
    me->setStatOverflows(kTRUE);
  }
  return me;
}

ElectronDqmAnalyzerBase::MonitorElement *ElectronDqmAnalyzerBase::bookH1withSumw2(DQMStore::IBooker &iBooker,
                                                                                  const std::string &name,
                                                                                  const std::string &title,
                                                                                  int nchX,
                                                                                  double lowX,
                                                                                  double highX,
                                                                                  const std::string &titleX,
                                                                                  const std::string &titleY,
                                                                                  Option_t *option) {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement *me = iBooker.book1D(newName(name), title, nchX, lowX, highX);
  if (me->getTH1F()->GetSumw2N() == 0)
    me->enableSumw2();
  if (!titleX.empty()) {
    me->setAxisTitle(titleX);
  }
  if (!titleY.empty()) {
    me->setAxisTitle(titleY, 2);
  }
  if (TString(option) != "") {
    me->setOption(option);
  }
  if (bookStatOverflowFlag_) {
    me->setStatOverflows(kTRUE);
  }
  return me;
}

ElectronDqmAnalyzerBase::MonitorElement *ElectronDqmAnalyzerBase::bookH2(DQMStore::IBooker &iBooker,
                                                                         const std::string &name,
                                                                         const std::string &title,
                                                                         int nchX,
                                                                         double lowX,
                                                                         double highX,
                                                                         int nchY,
                                                                         double lowY,
                                                                         double highY,
                                                                         const std::string &titleX,
                                                                         const std::string &titleY,
                                                                         Option_t *option) {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement *me = iBooker.book2D(newName(name), title, nchX, lowX, highX, nchY, lowY, highY);
  if (!titleX.empty()) {
    me->setAxisTitle(titleX);
  }
  if (!titleY.empty()) {
    me->setAxisTitle(titleY, 2);
  }
  if (TString(option) != "") {
    me->setOption(option);
  }
  if (bookStatOverflowFlag_) {
    me->setStatOverflows(kTRUE);
  }
  return me;
}

ElectronDqmAnalyzerBase::MonitorElement *ElectronDqmAnalyzerBase::bookH2withSumw2(DQMStore::IBooker &iBooker,
                                                                                  const std::string &name,
                                                                                  const std::string &title,
                                                                                  int nchX,
                                                                                  double lowX,
                                                                                  double highX,
                                                                                  int nchY,
                                                                                  double lowY,
                                                                                  double highY,
                                                                                  const std::string &titleX,
                                                                                  const std::string &titleY,
                                                                                  Option_t *option) {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement *me = iBooker.book2D(newName(name), title, nchX, lowX, highX, nchY, lowY, highY);
  if (me->getTH2F()->GetSumw2N() == 0)
    me->enableSumw2();
  if (!titleX.empty()) {
    me->setAxisTitle(titleX);
  }
  if (!titleY.empty()) {
    me->setAxisTitle(titleY, 2);
  }
  if (TString(option) != "") {
    me->setOption(option);
  }
  if (bookStatOverflowFlag_) {
    me->setStatOverflows(kTRUE);
  }
  return me;
}

ElectronDqmAnalyzerBase::MonitorElement *ElectronDqmAnalyzerBase::bookP1(DQMStore::IBooker &iBooker,
                                                                         const std::string &name,
                                                                         const std::string &title,
                                                                         int nchX,
                                                                         double lowX,
                                                                         double highX,
                                                                         double lowY,
                                                                         double highY,
                                                                         const std::string &titleX,
                                                                         const std::string &titleY,
                                                                         Option_t *option) {
  iBooker.setCurrentFolder(outputInternalPath_);
  MonitorElement *me = iBooker.bookProfile(newName(name), title, nchX, lowX, highX, lowY, highY, " ");
  if (!titleX.empty()) {
    me->getTProfile()->GetXaxis()->SetTitle(titleX.c_str());
  }
  if (!titleY.empty()) {
    me->setAxisTitle(titleY, 2);
  }
  if (TString(option) != "") {
    me->getTProfile()->SetOption(option);
  }
  if (bookStatOverflowFlag_) {
    me->setStatOverflows(kTRUE);
  }
  return me;
}
