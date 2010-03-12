#ifndef EBDataCertificationTask_h
#define EBDataCertificationTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TH2F.h"

class EBDataCertificationTask: public edm::EDAnalyzer{

public:

/// Constructor
EBDataCertificationTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBDataCertificationTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(void);

/// EndJob
void endJob(void);

/// BeginLuminosityBlock
void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup);

/// EndLuminosityBlock
void endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup);

/// Reset
void reset(void);

/// Cleanup
void cleanup(void);
  
private:

bool cloneME_;
  
DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

TH2F *hDQM_;
TH2F *hDAQ_;
TH2F *hDCS_;
TH1F *hIntegrityByLumi_;
TH1F *hFrontendByLumi_;
TH1F *hDCSByLumi_;

MonitorElement* meEBDataCertificationSummary_;
MonitorElement* meEBDataCertification_[36];
MonitorElement* meEBDataCertificationSummaryMap_;

MonitorElement* meDataQualityByLumi_;
MonitorElement* meDCSQualityByLumi_;

};

#endif
