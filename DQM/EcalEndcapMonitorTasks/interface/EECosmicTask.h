#ifndef EECosmicTask_H
#define EECosmicTask_H

/*
 * \file EECosmicTask.h
 *
 * $Date: 2007/03/20 12:37:26 $
 * $Revision: 1.21 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EECosmicTask: public edm::EDAnalyzer{

public:

/// Constructor
EECosmicTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EECosmicTask();

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

edm::InputTag EcalRawDataCollection_;
edm::InputTag EcalRecHitCollection_;

MonitorElement* meCutMap_[36];

MonitorElement* meSelMap_[36];

MonitorElement* meSpectrumMap_[36];

bool init_;

};

#endif
