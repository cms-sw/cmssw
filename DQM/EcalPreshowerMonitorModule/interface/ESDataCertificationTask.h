#ifndef ESDataCertificationTask_h
#define ESDataCertificationTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

class ESDataCertificationTask : public edm::EDAnalyzer {
public:
  typedef dqm::legacy::MonitorElement MonitorElement;
  typedef dqm::legacy::DQMStore DQMStore;

  ESDataCertificationTask(const edm::ParameterSet& ps);
  ~ESDataCertificationTask() override;

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void beginJob(void) override;
  void endJob(void) override;
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& iSetup) override;
  void reset(void);

private:
  DQMStore* dqmStore_;

  std::string prefixME_;

  bool mergeRuns_;

  MonitorElement* meESDataCertificationSummary_;
  MonitorElement* meESDataCertificationSummaryMap_;
};

#endif
