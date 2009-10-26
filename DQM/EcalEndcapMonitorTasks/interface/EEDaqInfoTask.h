#ifndef EEDaqInfoTask_h
#define EEDaqInfoTask_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EEDaqInfoTask: public edm::EDAnalyzer{

public:

/// Constructor
EEDaqInfoTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEDaqInfoTask();

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
  
DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

MonitorElement* meEEDaqFraction_;
MonitorElement* meEEDaqActive_[18];
MonitorElement* meEEDaqActiveMap_;

int EEMinusFedRangeMin_;
int EEMinusFedRangeMax_;
int EEPlusFedRangeMin_;
int EEPlusFedRangeMax_;

};

#endif
