#ifndef DQMServices_Core_DQMEDAnalyzer_h
#define DQMServices_Core_DQMEDAnalyzer_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Utilities/interface/EDPutToken.h"
#include "DataFormats/Histograms/interface/DQMToken.h"

class DQMEDAnalyzer : public edm::one::EDProducer<edm::Accumulator,
                                                  edm::EndLuminosityBlockProducer,
                                                  edm::EndRunProducer,
                                                  edm::one::WatchLuminosityBlocks,
                                                  edm::one::WatchRuns> 
{
public:
  DQMEDAnalyzer();
  ~DQMEDAnalyzer() override = default;
  DQMEDAnalyzer(DQMEDAnalyzer const&) = delete;
  DQMEDAnalyzer(DQMEDAnalyzer &&) = delete;

  void beginRun(edm::Run const& run, edm::EventSetup const& setup) final;

  void endRun(edm::Run const& run, edm::EventSetup const& setup) override;
  void endRunProduce(edm::Run& run, edm::EventSetup const& setup) override;

  void beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) override;

  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void endLuminosityBlockProduce(edm::LuminosityBlock & lumi, edm::EventSetup const& setup) final;


  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&);
  virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) = 0;

  virtual void analyze(edm::Event const&, edm::EventSetup const&);
  void accumulate(edm::Event const& ev, edm::EventSetup const& es) final;

protected:
  edm::EDPutTokenT<DQMToken> lumiToken_;
  edm::EDPutTokenT<DQMToken> runToken_;

};

#endif // DQMServices_Core_DQMEDAnalyzer_h
