#ifndef EBTriggerTowerTask_H
#define EBTriggerTowerTask_H

/*
 * \file EBTriggerTowerTask.h
 *
 * $Date: 2007/03/20 12:37:26 $
 * $Revision: 1.5 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EBTriggerTowerTask: public edm::EDAnalyzer{

public:

/// Constructor
EBTriggerTowerTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBTriggerTowerTask();

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

edm::InputTag EcalTrigPrimDigiCollection_;
edm::InputTag EcalUncalibratedRecHitCollection_;

MonitorElement* meEtMap_[36];

MonitorElement* meVeto_[36];

MonitorElement* meFlags_[36];

MonitorElement* meEtMapT_[36][68];
MonitorElement* meEtMapR_[36][68];

bool init_;

};

#endif
