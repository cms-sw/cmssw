#ifndef EBDaqInfoTask_h
#define EBDaqInfoTask_h

/*
 * \file EBDaqInfoTask.h
 *
 * $Date: 2010/08/08 08:56:00 $
 * $Revision: 1.4 $
 * \author E. Di Marco
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EBDaqInfoTask: public edm::EDAnalyzer{

public:

/// Constructor
EBDaqInfoTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBDaqInfoTask();

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

void fillMonitorElements(int ready[72][34]);
  
DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

MonitorElement* meEBDaqFraction_;
MonitorElement* meEBDaqActive_[36];
MonitorElement* meEBDaqActiveMap_;

int readyRun[72][34];
int readyLumi[72][34];

};

#endif
