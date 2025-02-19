#ifndef EBIntegrityTask_H
#define EBIntegrityTask_H

/*
 * \file EBIntegrityTask.h
 *
 * $Date: 2012/04/27 13:46:00 $
 * $Revision: 1.33 $
 * \author G. Della Ricca
 *
 */


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

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

MonitorElement* meIntegrityChId[36];
MonitorElement* meIntegrityGain[36];
MonitorElement* meIntegrityGainSwitch[36];
MonitorElement* meIntegrityTTId[36];
MonitorElement* meIntegrityTTBlockSize[36];
MonitorElement* meIntegrityMemChId[36];
MonitorElement* meIntegrityMemGain[36];
MonitorElement* meIntegrityMemTTId[36];
MonitorElement* meIntegrityMemTTBlockSize[36];
MonitorElement* meIntegrityDCCSize;
MonitorElement* meIntegrityErrorsByLumi;

bool init_;

const static int chMemAbscissa[25];
const static int chMemOrdinate[25];

};

#endif
