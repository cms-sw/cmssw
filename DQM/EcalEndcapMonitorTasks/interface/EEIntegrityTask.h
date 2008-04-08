#ifndef EEIntegrityTask_H
#define EEIntegrityTask_H

/*
 * \file EEIntegrityTask.h
 *
 * $Date: 2008/02/29 15:07:48 $
 * $Revision: 1.10 $
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

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dqmStore_;

bool enableCleanup_;

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
