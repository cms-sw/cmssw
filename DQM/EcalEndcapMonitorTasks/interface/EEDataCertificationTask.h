#ifndef EEDataCertificationTask_h
#define EEDataCertificationTask_h

/*
 * \file EEDataCertificationTask.h
 *
 * $Date: 2011/06/27 08:35:13 $
 * $Revision: 1.8 $
 * \author E. Di Marco
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TH2F.h"

class EEDataCertificationTask: public edm::EDAnalyzer{

public:

/// Constructor
EEDataCertificationTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEDataCertificationTask();

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

/// BeginRun
void beginRun(const edm::Run & r, const edm::EventSetup & c);

/// EndRun
void endRun(const edm::Run & r, const edm::EventSetup & c);

/// Reset
void reset(void);

/// Cleanup
void cleanup(void);

 static const int DccId_[18];
 static const int nTowerMax_;
  
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
TH1F *hSynchronizationByLumi_;

MonitorElement* meEEDataCertificationSummary_;
MonitorElement* meEEDataCertification_[18];
MonitorElement* meEEDataCertificationSummaryMap_;
MonitorElement* meEEReportSummary_;
MonitorElement* meEEReportSummaryContents_[36];

};

const int EEDataCertificationTask::DccId_[18] = {1,2,3,4,5,6,7,8,9,
				       46,47,48,49,50,51,52,53,54};
const int EEDataCertificationTask::nTowerMax_ = 44;

#endif
