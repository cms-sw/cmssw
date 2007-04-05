#ifndef EBIntegrityTask_H
#define EBIntegrityTask_H

/*
 * \file EBIntegrityTask.h
 *
 * $Date: 2007/04/05 13:56:46 $
 * $Revision: 1.16 $
 * \author G. Della Ricca
 *
 */


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EBIntegrityTask: public edm::EDAnalyzer{

public:

/// Constructor
EBIntegrityTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBIntegrityTask();

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

DaqMonitorBEInterface* dbe_;

bool enableCleanup_;

edm::InputTag EBDetIdCollection0_;
edm::InputTag EBDetIdCollection1_;
edm::InputTag EBDetIdCollection2_;
edm::InputTag EBDetIdCollection3_;
edm::InputTag EBDetIdCollection4_;
edm::InputTag EcalTrigTowerDetIdCollection1_;
edm::InputTag EcalTrigTowerDetIdCollection2_;
edm::InputTag EcalElectronicsIdCollection1_;
edm::InputTag EcalElectronicsIdCollection2_;
edm::InputTag EcalElectronicsIdCollection3_;
edm::InputTag EcalElectronicsIdCollection4_;

MonitorElement* meIntegrityChId[36];
MonitorElement* meIntegrityGain[36];
MonitorElement* meIntegrityGainSwitch[36];
MonitorElement* meIntegrityGainSwitchStay[36];
MonitorElement* meIntegrityTTId[36];
MonitorElement* meIntegrityTTBlockSize[36];
MonitorElement* meIntegrityMemChId[36];
MonitorElement* meIntegrityMemGain[36];
MonitorElement* meIntegrityMemTTId[36];
MonitorElement* meIntegrityMemTTBlockSize[36];
MonitorElement* meIntegrityDCCSize;

bool init_;

const static int chMemAbscissa[25];
const static int chMemOrdinate[25];

};

#endif
