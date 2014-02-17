#ifndef EBStatusFlagsTask_H
#define EBStatusFlagsTask_H

/*
 * \file EBStatusFlagsTask.h
 *
 * $Date: 2012/04/27 13:46:00 $
 * $Revision: 1.12 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBStatusFlagsTask: public edm::EDAnalyzer{

public:

/// Constructor
EBStatusFlagsTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBStatusFlagsTask();

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

edm::InputTag EcalRawDataCollection_;

MonitorElement* meEvtType_[36];

MonitorElement* meFEchErrors_[36][3];
MonitorElement* meFEchErrorsByLumi_;

bool init_;

};

#endif
