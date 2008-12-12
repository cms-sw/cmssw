#ifndef EBDaqInfoTask_h
#define EBDaqInfoTask_h

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
void beginJob(const edm::EventSetup& c);

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

MonitorElement* meEBDaqFraction_;
MonitorElement* meEBDaqActive_[36];
MonitorElement* meEBDaqActiveMap_;

int EBFedRangeMin_;
int EBFedRangeMax_;

};

#endif
