#ifndef SiPixel_DefaultTemplates_h
#define SiPixel_DefaultTemplates_h
// 
// This does not really define any class; this file has two templates that 
// expand into DQM framework plugins using one base implementation.
//
// As with the entire SiPixelPhase1Common framework, you do not have to use
// this but can still use the other helpers. However, the HistogramManager
// needs to run in step1 and step2 of the DQM, and need to have exectly the
// same spec to work properly. This is why we play some template tricks here
// to generate a Harvester and an Analyzer from the same code.
// The cost is a lot of hazzle with the consumes<...>. Look at some example
// code to get it right.
//
// To use the templates use sth. like this:
// typedef SiPixelPhase1Analyzer<SiPixelPhase1Digis> SiPixelPhase1DigisAnalyzer;
// typedef SiPixelPhase1Harvester<SiPixelPhase1Digis> SiPixelPhase1DigisHarvester;
// DEFINE_FWK_MODULE(SiPixelPhase1DigisAnalyzer);
// DEFINE_FWK_MODULE(SiPixelPhase1DigisHarvester);
//
// Original Author: Marcel Schneider
//

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

#include <vector>

// This is the base class your plugin may derive from. You are not required to
// use it (and can still use the templates below it) but if you just need some
// normal HistogramManager this should be perfect.
// Note that this does not derive from FWCore, even though the interface looks
// a lot like a FWCore plugin.
class SiPixelPhase1Base {
  public:
  
  // You should add a constructor that sets up the specs.
  SiPixelPhase1Base(const edm::ParameterSet& iConfig, int NQuantities = 1) {
    for (int i = 0; i < NQuantities; i++) 
      histo.emplace_back(HistogramManager(iConfig));
  };
  
  // You should analyze something, and call histoman.fill(...).
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup);

  // Note that for the Analyzer template, you need to define sth. like this:
  //template<class Consumer>
  //void SiPixelPhase1Digis::registerConsumes(const edm::ParameterSet& iConfig, Consumer& c) {
  //    srcToken_ = c.template consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("src"));
  //}
  // We cannot give a default in this base class. Also note the weird syntax with consumes, g++ wants it like this.

  // These are directly called from the corresponding DQM framework methods. 
  // If you just use HistogramManager, the default impls are just fine.
  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& iSetup) {
    for (HistogramManager& histoman : histo)
      histoman.book(iBooker, iSetup);
  };
  void dqmEndLuminosityBlock(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup) {
    for (HistogramManager& histoman : histo)
      histoman.executeHarvestingOnline(iBooker, iGetter, eSetup);
  };
  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
    for (HistogramManager& histoman : histo)
      histoman.executeHarvestingOffline(iBooker, iGetter);
  };

  protected:
  std::vector<HistogramManager> histo;
};

// This wraps the SiPixelPhase1Base (or anything else that has the right methods) 
// into an DQMEDAnalyzer. In addition to the usual methods it calls a template 
// registerConsumes where all the consumes<...> calls have to be performed; see
// comment above.
template<class Impl>
class SiPixelPhase1Analyzer : public DQMEDAnalyzer {
  public: 
  SiPixelPhase1Analyzer(const edm::ParameterSet& ps) : impl(ps) {
    impl.registerConsumes(ps, *this);
  };

  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& iSetup) {
    impl.bookHistograms(iBooker, run, iSetup);
  };
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) {
    impl.analyze(e, eSetup);
  };

  // by default this is protected, this makes it visible
  template <typename ProductType, edm::BranchType B=edm::InEvent>
  edm::EDGetTokenT<ProductType> consumes(edm::InputTag const& tag) {
    return edm::EDConsumerBase::consumes<ProductType, B>(tag);
  };

  private:
  Impl impl;
};

// This wraps the SiPixelPhase1Base into a DQMEDHarvester. SiPixelPhase1Base
// provides sane default implementations, so most plugins don't care about this.
template<class Impl> 
class SiPixelPhase1Harvester : public DQMEDHarvester {
  friend class IConsumer;
  public: 
  SiPixelPhase1Harvester(const edm::ParameterSet& ps) : impl(ps) {};

  void dqmEndLuminosityBlock(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup) {
    impl.dqmEndLuminosityBlock(iBooker, iGetter, lumiBlock, eSetup);
  };
  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
    impl.dqmEndJob(iBooker, iGetter);
  };

  private:
  Impl impl;
};

#endif
