#ifndef EBPedestalOnlineTask_H
#define EBPedestalOnlineTask_H

/*
 * \file EBPedestalOnlineTask.h
 *
 * $Date: 2008/04/08 15:06:23 $
 * $Revision: 1.16 $
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

edm::InputTag EBDigiCollection_;

MonitorElement* mePedMapG12_[36];

bool init_;

};

#endif
