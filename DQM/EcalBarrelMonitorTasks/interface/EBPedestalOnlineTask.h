#ifndef EBPedestalOnlineTask_H
#define EBPedestalOnlineTask_H

/*
 * \file EBPedestalOnlineTask.h
 *
 * $Date: 2008/05/11 09:35:08 $
 * $Revision: 1.18 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EBPedestalOnlineTask: public edm::EDAnalyzer{

public:

/// Constructor
EBPedestalOnlineTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBPedestalOnlineTask();

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

edm::InputTag EBDigiCollection_;

MonitorElement* mePedMapG12_[36];

bool init_;

};

#endif
