#ifndef ESDcsInfoTask_h
#define ESDcsInfoTask_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DQMServices/Core/interface/DQMStore.h"

class ESDcsInfoTask : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchLuminosityBlocks> {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  /// Constructor
  ESDcsInfoTask(const edm::ParameterSet& ps);

  /// Destructor
  ~ESDcsInfoTask() override;

protected:
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  /// BeginJob
  void beginJob(void) override;

  /// EndJob
  void endJob(void) override;

  /// BeginLuminosityBlock
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& iSetup) override;

  /// EndLuminosityBlock
  void endLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;

  /// Reset
  void reset(void);

private:
  DQMStore* dqmStore_;

  std::string prefixME_;

  bool mergeRuns_;

  edm::EDGetTokenT<DcsStatusCollection> dcsStatustoken_;

  MonitorElement* meESDcsFraction_;
  MonitorElement* meESDcsActiveMap_;

  int ievt_;
};

#endif
