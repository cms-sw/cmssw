#ifndef EEPedestalOnlineTask_H
#define EEPedestalOnlineTask_H

/*
 * \file EEPedestalOnlineTask.h
 *
 * $Date: 2007/04/02 16:23:12 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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

DaqMonitorBEInterface* dbe_;

edm::InputTag EBDigiCollection_;

MonitorElement* mePedMapG12_[36];

bool init_;

};

#endif
