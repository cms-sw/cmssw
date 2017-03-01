#ifndef SiPixel_DefaultTemplates_h
#define SiPixel_DefaultTemplates_h
// 
// This defines two classes, one that has to be extended to make a new plugin,
// and one that can be used as-is for the Harvesting.
//
// As with the entire SiPixelPhase1Common framework, you do not have to use
// this but can still use the other helpers. However, the HistogramManager
// needs to run in step1 and step2 of the DQM, and need to have exactly the
// same spec to work properly. This has to be guranteed by the configuration.
//
// Original Author: Marcel Schneider
//

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

#include <vector>

// used as a mixin for Analyzer and Harvester.
class HistogramManagerHolder {
  public:
  HistogramManagerHolder(const edm::ParameterSet& iConfig)
    : geometryInterface(iConfig.getParameter<edm::ParameterSet>("geometry")) {
    auto histograms = iConfig.getParameter<edm::VParameterSet>("histograms");
    for (auto histoconf : histograms) {
      histo.emplace_back(HistogramManager(histoconf, geometryInterface));
    }
  };

  protected:
  std::vector<HistogramManager> histo;
  GeometryInterface geometryInterface;
};

// This is the base class your plugin may derive from. You are not required to
// use it but if you just need some normal HistogramManager this should be perfect.
class SiPixelPhase1Base : public DQMEDAnalyzer, public HistogramManagerHolder {
  public:
  SiPixelPhase1Base(const edm::ParameterSet& iConfig) 
    : DQMEDAnalyzer(), HistogramManagerHolder(iConfig) {};

  // You should analyze something, and call histoman.fill(...).
  //void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

  // This booking is usually fine.
  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& iSetup) {
    for (HistogramManager& histoman : histo)
      histoman.book(iBooker, iSetup);
  };

  virtual ~SiPixelPhase1Base() {};
};

// This wraps the Histogram Managers into a DQMEDHarvester. It 
// provides sane default implementations, so most plugins don't care about this.
// However, you have to instantiate one with the same config as your Analyzer 
// to get the Harvesting done.
// For custom harvesting, you have to derive from this.
class SiPixelPhase1Harvester : public DQMEDHarvester, public HistogramManagerHolder {
  public:
  SiPixelPhase1Harvester(const edm::ParameterSet& iConfig) 
    : DQMEDHarvester(), HistogramManagerHolder(iConfig) {};

  void dqmEndLuminosityBlock(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup) ;
  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter);
};
#endif
