#ifndef EEIntegrityTask_H
#define EEIntegrityTask_H

/*
 * \file EEIntegrityTask.h
 *
 * $Date: 2008/04/08 15:32:09 $
 * $Revision: 1.12 $
 * \author G. Della Ricca
 *
 */


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EEIntegrityTask: public edm::EDAnalyzer{

public:

/// Constructor
EEIntegrityTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEIntegrityTask();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

/// BeginJob
void beginJob(const edm::EventSetup& c);

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

edm::InputTag EEDetIdCollection0_;
edm::InputTag EEDetIdCollection1_;
edm::InputTag EEDetIdCollection2_;
edm::InputTag EEDetIdCollection3_;
edm::InputTag EEDetIdCollection4_;
edm::InputTag EcalElectronicsIdCollection1_;
edm::InputTag EcalElectronicsIdCollection2_;
edm::InputTag EcalElectronicsIdCollection3_;
edm::InputTag EcalElectronicsIdCollection4_;
edm::InputTag EcalElectronicsIdCollection5_;
edm::InputTag EcalElectronicsIdCollection6_;

MonitorElement* meIntegrityChId[18];
MonitorElement* meIntegrityGain[18];
MonitorElement* meIntegrityGainSwitch[18];
MonitorElement* meIntegrityTTId[18];
MonitorElement* meIntegrityTTBlockSize[18];
MonitorElement* meIntegrityMemChId[18];
MonitorElement* meIntegrityMemGain[18];
MonitorElement* meIntegrityMemTTId[18];
MonitorElement* meIntegrityMemTTBlockSize[18];
MonitorElement* meIntegrityDCCSize;

bool init_;

const static int chMemAbscissa[25];
const static int chMemOrdinate[25];

};

#endif
