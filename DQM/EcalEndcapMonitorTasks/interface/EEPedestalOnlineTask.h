#ifndef EEPedestalOnlineTask_H
#define EEPedestalOnlineTask_H

/*
 * \file EEPedestalOnlineTask.h
 *
 * $Date: 2008/04/08 15:06:27 $
 * $Revision: 1.9 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EEPedestalOnlineTask: public edm::EDAnalyzer{

public:

/// Constructor
EEPedestalOnlineTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEPedestalOnlineTask();

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

std::string prefixME_;

bool enableCleanup_;

edm::InputTag EEDigiCollection_;

MonitorElement* mePedMapG12_[18];

bool init_;

};

#endif
