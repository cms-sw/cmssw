#ifndef EEDaqInfoTask_h
#define EEDaqInfoTask_h

/*
 * \file EEDaqInfoTask.h
 *
 * $Date: 2010/08/08 08:56:00 $
 * $Revision: 1.4 $
 * \author E. Di Marco
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

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

 void fillMonitorElements(int ready[40][20], const EcalElectronicsMapping *);
  
DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

MonitorElement* meEEDaqFraction_;
MonitorElement* meEEDaqActive_[18];
MonitorElement* meEEDaqActiveMap_;

int readyRun[40][20];
int readyLumi[40][20];

};

const int EEDaqInfoTask::DccId_[18] = {1,2,3,4,5,6,7,8,9,
				       46,47,48,49,50,51,52,53,54};
const int EEDaqInfoTask::nTowerMax_ = 44;

#endif
