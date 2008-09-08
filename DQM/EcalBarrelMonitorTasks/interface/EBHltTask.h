#ifndef EBHltTask_H
#define EBHltTask_H

/*
 * \file EBHltTask.h
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

class EBHltTask: public edm::EDAnalyzer{

public:

/// Constructor
EBHltTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBHltTask();

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

edm::InputTag EBDetIdCollection0_;
edm::InputTag EBDetIdCollection1_;
edm::InputTag EBDetIdCollection2_;
edm::InputTag EBDetIdCollection3_;
edm::InputTag EBDetIdCollection4_;
edm::InputTag EcalElectronicsIdCollection1_;
edm::InputTag EcalElectronicsIdCollection2_;
edm::InputTag EcalElectronicsIdCollection3_;
edm::InputTag EcalElectronicsIdCollection4_;
edm::InputTag EcalElectronicsIdCollection5_;
edm::InputTag EcalElectronicsIdCollection6_;
edm::InputTag FEDRawDataCollection_;

MonitorElement* meEBFedsOccupancy_;
MonitorElement* meEBFedsSizeErrors_;
MonitorElement* meEBFedsIntegrityErrors_;

bool init_;

};

#endif
