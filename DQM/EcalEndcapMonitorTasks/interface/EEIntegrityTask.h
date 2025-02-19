#ifndef EEIntegrityTask_H
#define EEIntegrityTask_H

/*
 * \file EEIntegrityTask.h
 *
 * $Date: 2012/04/27 13:46:13 $
 * $Revision: 1.20 $
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

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

 std::string subfolder_;

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
MonitorElement* meIntegrityErrorsByLumi;

bool init_;

const static int chMemAbscissa[25];
const static int chMemOrdinate[25];

};

#endif
