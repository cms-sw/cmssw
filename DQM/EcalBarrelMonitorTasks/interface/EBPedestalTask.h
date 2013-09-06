#ifndef EBPedestalTask_H
#define EBPedestalTask_H

/*
 * \file EBPedestalTask.h
 *
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

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

edm::EDGetTokenT<EcalRawDataCollection> EcalRawDataCollection_;
edm::EDGetTokenT<EBDigiCollection> EBDigiCollection_;
edm::EDGetTokenT<EcalPnDiodeDigiCollection> EcalPnDiodeDigiCollection_;
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
