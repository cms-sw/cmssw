#ifndef EEHltTask_H
#define EEHltTask_H

/*
 * \file EEHltTask.h
 *
 * $Date: 2008/05/11 09:35:08 $
 * $Revision: 1.30 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EEHltTask: public edm::EDAnalyzer{

public:

/// Constructor
EEHltTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEHltTask();

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
edm::InputTag FEDRawDataCollection_;

MonitorElement* meEEFedsOccupancy_;
MonitorElement* meEEFedsSizeErrors_;
MonitorElement* meEEFedsIntegrityErrors_;

bool init_;

};

#endif
