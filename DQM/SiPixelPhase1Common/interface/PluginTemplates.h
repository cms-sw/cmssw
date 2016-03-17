#ifndef SiPixel_DefaultTemplates_h
#define SiPixel_DefaultTemplates_h
// 
// This does not really define any class; this file has two templates that 
// expand into DQM framework plugins using one base implementation.
//
// Original Author: Marcel Schneider
//

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"

#include "DQM/SiPixelPhase1Common/interface/HistogramManager.h"

//class IConsumer {
  //public:
  //IConsumer(edm::EDConsumerBase* a) : analyzer(a) {};
  //template <typename edm::ProductType, edm::BranchType B=edm::InEvent>
  //edm::EDGetTokenT<ProductType> consumes(edm::InputTag const& tag) {
    //return a->consumes<ProductType, B>(tag);
  //};
  //private:
  //edm::EDConsumerBase* a;
//};

class SiPixelPhase1Base {
  public:
  SiPixelPhase1Base(const edm::ParameterSet& ps) : histoman(ps) {};
  void bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& iSetup) {
    histoman.book(iBooker, iSetup);
  };
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) {};
  void dqmEndLuminosityBlock(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter, edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup) {
    histoman.executeHarvestingOnline(iBooker, iGetter, eSetup);
  };
  void dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
    histoman.executeHarvestingOffline(iBooker, iGetter);
  };

  protected:
  HistogramManager histoman;
};

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
  template <typename ProductType, edm::BranchType B=edm::InEvent>
  edm::EDGetTokenT<ProductType> consumes(edm::InputTag const& tag) {
    return edm::EDConsumerBase::consumes<ProductType, B>(tag);
  };


  private:
  Impl impl;
};

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
