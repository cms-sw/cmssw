#ifndef EBCosmicTask_H
#define EBCosmicTask_H

/*
 * \file EBCosmicTask.h
 *
 * $Date: 2007/04/05 13:56:46 $
 * $Revision: 1.22 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EBCosmicTask: public edm::EDAnalyzer{

public:

/// Constructor
EBCosmicTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBCosmicTask();

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

edm::InputTag EcalRawDataCollection_;
edm::InputTag EcalRecHitCollection_;

MonitorElement* meCutMap_[36];

MonitorElement* meSelMap_[36];

MonitorElement* meSpectrumMap_[36];

bool init_;

};

#endif
