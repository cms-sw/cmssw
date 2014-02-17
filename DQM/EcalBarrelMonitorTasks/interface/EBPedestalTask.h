#ifndef EBPedestalTask_H
#define EBPedestalTask_H

/*
 * \file EBPedestalTask.h
 *
 * $Date: 2012/04/27 13:46:00 $
 * $Revision: 1.35 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBPedestalTask: public edm::EDAnalyzer{

public:

/// Constructor
EBPedestalTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBPedestalTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(void);

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(const edm::Run & r, const edm::EventSetup & c);

/// EndRun
void endRun(const edm::Run & r, const edm::EventSetup & c);

/// Reset
void reset(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

edm::InputTag EcalRawDataCollection_;
edm::InputTag EBDigiCollection_;
edm::InputTag EcalPnDiodeDigiCollection_;
std::vector<int> MGPAGains_;
std::vector<int> MGPAGainsPN_;

MonitorElement* mePedMapG01_[36];
MonitorElement* mePedMapG06_[36];
MonitorElement* mePedMapG12_[36];

MonitorElement* mePed3SumMapG01_[36];
MonitorElement* mePed3SumMapG06_[36];
MonitorElement* mePed3SumMapG12_[36];

MonitorElement* mePed5SumMapG01_[36];
MonitorElement* mePed5SumMapG06_[36];
MonitorElement* mePed5SumMapG12_[36];

MonitorElement* mePnPedMapG01_[36];
MonitorElement* mePnPedMapG16_[36];

bool init_;

};

#endif
