#ifndef EETriggerTowerTask_H
#define EETriggerTowerTask_H

/*
 * \file EETriggerTowerTask.h
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

class EETriggerTowerTask: public edm::EDAnalyzer{

public:

/// Constructor
EETriggerTowerTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EETriggerTowerTask();

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
