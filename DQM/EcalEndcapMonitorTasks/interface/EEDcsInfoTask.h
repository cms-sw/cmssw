#ifndef EEDcsInfoTask_h
#define EEDcsInfoTask_h

/*
 * \file EEDcsInfoTask.h
 *
 * $Date: 2010/08/08 08:46:05 $
 * $Revision: 1.132 $
 * \author E. Di Marco
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EEDcsInfoTask: public edm::EDAnalyzer{

public:

/// Constructor
EEDcsInfoTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEDcsInfoTask();

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
  
private:

void fillMonitorElements(int ready[40][20]);
  
DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

MonitorElement* meEEDcsFraction_;
MonitorElement* meEEDcsActive_[18];
MonitorElement* meEEDcsActiveMap_;

int readyRun[40][20];
int readyLumi[40][20];

};

#endif
