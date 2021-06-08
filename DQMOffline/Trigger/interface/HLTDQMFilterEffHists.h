#ifndef DQMOnline_Trigger_HLTDQMHistColl_h
#define DQMOnline_Trigger_HLTDQMHistColl_h

//********************************************************************************
//
// Description:
//   This contains a collection of HLTDQMHists used to measure the efficiency of a
//   specified filter. It is resonsible for booking and filling the histograms
//   For every hist specified, it books two, one to record the
//   total objects passing sample selection passed to the class and then one for those objects
//   which then pass the HLT filter
//   The class contains a simple selection cuts (mostly intended for kinematic range cuts)
//   which are passed to the histograms as they are filled.
//   The cuts are passed to the histograms as some histograms will ignore certain cuts
//   for example, eff vs Et will ignore any global Et cuts applied
//
// Author : Sam Harper , RAL, May 2017
//
//***********************************************************************************

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DQMOffline/Trigger/interface/HLTDQMHist.h"
#include "DQMOffline/Trigger/interface/VarRangeCutColl.h"
#include "DQMOffline/Trigger/interface/FunctionDefs.h"
#include "DQMOffline/Trigger/interface/UtilFuncs.h"

template <typename ObjType>
class HLTDQMFilterEffHists {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  explicit HLTDQMFilterEffHists(const edm::ParameterSet& config, std::string baseHistName, std::string hltProcess);

  static edm::ParameterSetDescription makePSetDescription();
  static edm::ParameterSetDescription makePSetDescriptionHistConfigs();

  void bookHists(DQMStore::IBooker& iBooker, const std::vector<edm::ParameterSet>& histConfigs);
  void fillHists(const ObjType& obj,
                 const edm::Event& event,
                 const edm::EventSetup& setup,
                 const trigger::TriggerEvent& trigEvt);

private:
  void book1D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig);
  void book2D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig);

private:
  std::vector<std::unique_ptr<HLTDQMHist<ObjType> > > histsPass_;
  std::vector<std::unique_ptr<HLTDQMHist<ObjType> > > histsTot_;
  VarRangeCutColl<ObjType> rangeCuts_;
  std::string filterName_;
  std::string histTitle_;
  std::string folderName_;
  std::string baseHistName_;
  std::string hltProcess_;
};

template <typename ObjType>
HLTDQMFilterEffHists<ObjType>::HLTDQMFilterEffHists(const edm::ParameterSet& config,
                                                    std::string baseHistName,
                                                    std::string hltProcess)
    : rangeCuts_(config.getParameter<std::vector<edm::ParameterSet> >("rangeCuts")),
      filterName_(config.getParameter<std::string>("filterName")),
      histTitle_(config.getParameter<std::string>("histTitle")),
      folderName_(config.getParameter<std::string>("folderName")),
      baseHistName_(std::move(baseHistName)),
      hltProcess_(std::move(hltProcess)) {}

template <typename ObjType>
edm::ParameterSetDescription HLTDQMFilterEffHists<ObjType>::makePSetDescription() {
  edm::ParameterSetDescription desc;
  desc.addVPSet("rangeCuts", VarRangeCut<ObjType>::makePSetDescription(), std::vector<edm::ParameterSet>());
  desc.add<std::string>("filterName", "");
  desc.add<std::string>("histTitle", "");
  desc.add<std::string>("folderName", "");
  return desc;
}

template <typename ObjType>
edm::ParameterSetDescription HLTDQMFilterEffHists<ObjType>::makePSetDescriptionHistConfigs() {
  edm::ParameterSetDescription desc;

  //what this is doing is trival and is left as an exercise to the reader
  auto histDescCases =
      "1D" >> (edm::ParameterDescription<std::vector<double> >("binLowEdges", std::vector<double>(), true) and
               edm::ParameterDescription<std::string>("nameSuffex", "", true) and
               edm::ParameterDescription<std::string>("vsVar", "", true)) or
      "2D" >> (edm::ParameterDescription<std::vector<double> >("xBinLowEdges", std::vector<double>(), true) and
               edm::ParameterDescription<std::vector<double> >("yBinLowEdges", std::vector<double>(), true) and
               edm::ParameterDescription<std::string>("nameSuffex", "", true) and
               edm::ParameterDescription<std::string>("xVar", "", true) and
               edm::ParameterDescription<std::string>("yVar", "", true));

  desc.ifValue(edm::ParameterDescription<std::string>("histType", "1D", true), std::move(histDescCases));
  desc.addVPSet("rangeCuts", VarRangeCut<ObjType>::makePSetDescription(), std::vector<edm::ParameterSet>());
  return desc;
}

template <typename ObjType>
void HLTDQMFilterEffHists<ObjType>::bookHists(DQMStore::IBooker& iBooker,
                                              const std::vector<edm::ParameterSet>& histConfigs) {
  iBooker.setCurrentFolder(folderName_);
  for (auto& histConfig : histConfigs) {
    std::string histType = histConfig.getParameter<std::string>("histType");
    if (histType == "1D") {
      book1D(iBooker, histConfig);
    } else if (histType == "2D") {
      book2D(iBooker, histConfig);
    } else {
      throw cms::Exception("ConfigError") << " histType " << histType << " not recognised" << std::endl;
    }
  }
}

template <typename ObjType>
void HLTDQMFilterEffHists<ObjType>::book1D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig) {
  auto binLowEdgesDouble = histConfig.getParameter<std::vector<double> >("binLowEdges");
  std::vector<float> binLowEdges;
  binLowEdges.reserve(binLowEdgesDouble.size());
  for (double lowEdge : binLowEdgesDouble)
    binLowEdges.push_back(lowEdge);
  auto nameSuffex = histConfig.getParameter<std::string>("nameSuffex");
  auto mePass = iBooker.book1D((baseHistName_ + filterName_ + nameSuffex + "_pass").c_str(),
                               (histTitle_ + nameSuffex + " Pass").c_str(),
                               binLowEdges.size() - 1,
                               &binLowEdges[0]);
  std::unique_ptr<HLTDQMHist<ObjType> > hist;
  auto vsVar = histConfig.getParameter<std::string>("vsVar");
  auto vsVarFunc = hltdqm::getUnaryFuncFloat<ObjType>(vsVar);
  if (!vsVarFunc) {
    throw cms::Exception("ConfigError") << " vsVar " << vsVar << " is giving null ptr (likely empty) in " << __FILE__
                                        << "," << __LINE__ << std::endl;
  }
  VarRangeCutColl<ObjType> rangeCuts(histConfig.getParameter<std::vector<edm::ParameterSet> >("rangeCuts"));
  hist = std::make_unique<HLTDQMHist1D<ObjType, float> >(mePass->getTH1(), vsVar, vsVarFunc, rangeCuts);
  histsPass_.emplace_back(std::move(hist));
  auto meTot = iBooker.book1D((baseHistName_ + filterName_ + nameSuffex + "_tot").c_str(),
                              (histTitle_ + nameSuffex + " Total").c_str(),
                              binLowEdges.size() - 1,
                              &binLowEdges[0]);
  hist = std::make_unique<HLTDQMHist1D<ObjType, float> >(meTot->getTH1(), vsVar, vsVarFunc, rangeCuts);
  histsTot_.emplace_back(std::move(hist));
}

template <typename ObjType>
void HLTDQMFilterEffHists<ObjType>::book2D(DQMStore::IBooker& iBooker, const edm::ParameterSet& histConfig) {
  auto xBinLowEdgesDouble = histConfig.getParameter<std::vector<double> >("xBinLowEdges");
  auto yBinLowEdgesDouble = histConfig.getParameter<std::vector<double> >("yBinLowEdges");
  std::vector<float> xBinLowEdges;
  std::vector<float> yBinLowEdges;
  xBinLowEdges.reserve(xBinLowEdgesDouble.size());
  for (double lowEdge : xBinLowEdgesDouble)
    xBinLowEdges.push_back(lowEdge);
  yBinLowEdges.reserve(yBinLowEdgesDouble.size());
  for (double lowEdge : yBinLowEdgesDouble)
    yBinLowEdges.push_back(lowEdge);
  auto nameSuffex = histConfig.getParameter<std::string>("nameSuffex");
  auto mePass = iBooker.book2D((baseHistName_ + filterName_ + nameSuffex + "_pass").c_str(),
                               (histTitle_ + nameSuffex + " Pass").c_str(),
                               xBinLowEdges.size() - 1,
                               &xBinLowEdges[0],
                               yBinLowEdges.size() - 1,
                               &yBinLowEdges[0]);
  std::unique_ptr<HLTDQMHist<ObjType> > hist;
  auto xVar = histConfig.getParameter<std::string>("xVar");
  auto yVar = histConfig.getParameter<std::string>("yVar");
  auto xVarFunc = hltdqm::getUnaryFuncFloat<ObjType>(xVar);
  auto yVarFunc = hltdqm::getUnaryFuncFloat<ObjType>(yVar);
  if (!xVarFunc || !yVarFunc) {
    throw cms::Exception("ConfigError") << " xVar " << xVar << " or yVar " << yVar
                                        << " is giving null ptr (likely empty str passed)" << std::endl;
  }
  VarRangeCutColl<ObjType> rangeCuts(histConfig.getParameter<std::vector<edm::ParameterSet> >("rangeCuts"));

  //really? really no MonitorElement::getTH2...sigh
  hist = std::make_unique<HLTDQMHist2D<ObjType, float> >(
      static_cast<TH2*>(mePass->getTH1()), xVar, yVar, xVarFunc, yVarFunc, rangeCuts);
  histsPass_.emplace_back(std::move(hist));

  auto meTot = iBooker.book2D((baseHistName_ + filterName_ + nameSuffex + "_tot").c_str(),
                              (histTitle_ + nameSuffex + " Total").c_str(),
                              xBinLowEdges.size() - 1,
                              &xBinLowEdges[0],
                              yBinLowEdges.size() - 1,
                              &yBinLowEdges[0]);

  hist = std::make_unique<HLTDQMHist2D<ObjType, float> >(
      static_cast<TH2*>(meTot->getTH1()), xVar, yVar, xVarFunc, yVarFunc, rangeCuts);
  histsTot_.emplace_back(std::move(hist));
}

template <typename ObjType>
void HLTDQMFilterEffHists<ObjType>::fillHists(const ObjType& obj,
                                              const edm::Event& event,
                                              const edm::EventSetup& setup,
                                              const trigger::TriggerEvent& trigEvt) {
  for (auto& hist : histsTot_) {
    hist->fill(obj, event, setup, rangeCuts_);
  }

  if (hltdqm::passTrig(obj.eta(), obj.phi(), trigEvt, filterName_, hltProcess_)) {
    for (auto& hist : histsPass_) {
      hist->fill(obj, event, setup, rangeCuts_);
    }
  }
}
#endif
