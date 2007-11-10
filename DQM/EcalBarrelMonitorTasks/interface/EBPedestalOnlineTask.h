#ifndef EBPedestalOnlineTask_H
#define EBPedestalOnlineTask_H

/*
 * \file EBPedestalOnlineTask.h
 *
 * $Date: 2007/04/05 14:53:59 $
 * $Revision: 1.12 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
class DaqMonitorBEInterface;

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

DaqMonitorBEInterface* dbe_;

bool enableCleanup_;

edm::InputTag EBDigiCollection_;

MonitorElement* mePedMapG12_[36];

bool init_;

};

#endif
