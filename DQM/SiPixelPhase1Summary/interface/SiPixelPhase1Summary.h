#ifndef SiPixelPhase1Summary_SiPixelPhase1Summary_h
#define SiPixelPhase1Summary_SiPixelPhase1Summary_h
// -*- C++ -*-
//
// Package:     SiPixelPhase1Summary
// Class  :     SiPixelPhase1Summary
//
/**

 Description: Summary map generation for the Phase 1 pixel

 Usage:
    <usage>

*/
//
// Original Author:  Duncan Leggat
//         Created:  2nd December 2016
//

//#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiPixelPhase1Summary : public DQMEDHarvester {
public:
  explicit SiPixelPhase1Summary(const edm::ParameterSet& conf);
  ~SiPixelPhase1Summary() override;

  //       virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //void dqmBeginRun(const edm::Run&, edm::EventSetup const&) ;
  //virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
protected:
  void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;

  void dqmEndLuminosityBlock(DQMStore::IBooker& iBooker,
                             DQMStore::IGetter& iGetter,
                             edm::LuminosityBlock const& lumiSeg,
                             edm::EventSetup const& c) override;
  //(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) override;
  //       virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  std::string topFolderName_;
  bool runOnEndLumi_;
  bool runOnEndJob_;

private:
  enum trendPlots { offline, fpix, bpix, layer1, layer2, layer3, layer4, ring1, ring2 };
  edm::ParameterSet conf_;
  edm::InputTag src_;
  bool firstLumi;

  std::map<std::string, MonitorElement*> summaryMap_;
  MonitorElement* deadROCSummary;
  MonitorElement* reportSummary;  //Float value of the average of the ins in the grand summary

  std::map<std::string, std::string> summaryPlotName_;

  //The dead and innefficient roc trend plot
  std::map<trendPlots, MonitorElement*> deadROCTrends_;
  std::map<trendPlots, MonitorElement*> ineffROCTrends_;

  //Error thresholds for the dead ROC plots
  std::vector<double> deadRocThresholds_;

  //book the summary plots
  void bookSummaries(DQMStore::IBooker& iBooker);

  //Book trend plots
  void bookTrendPlots(DQMStore::IBooker& iBooker);

  void fillSummaries(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter);

  void fillTrendPlots(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, int lumiSeg = 0);
};

#endif
