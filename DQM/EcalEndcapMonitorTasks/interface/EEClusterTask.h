#ifndef EEClusterTask_H
#define EEClusterTask_H

/*
 * \file EEClusterTask.h
 *
 * $Date: 2007/04/05 13:56:48 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class EEClusterTask: public edm::EDAnalyzer{

 public:

/// Constructor
EEClusterTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEClusterTask();

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

edm::InputTag islandBarrelBasicClusterCollection_;
edm::InputTag islandBarrelSuperClusterCollection_;

edm::InputTag hybridSuperClusterCollection_;

edm::InputTag hybridBarrelClusterShapeAssociation_;

MonitorElement* meIslBEne_;
MonitorElement* meIslBNum_;
MonitorElement* meIslBCry_;

MonitorElement* meIslBEneMap_;
MonitorElement* meIslBNumMap_;
MonitorElement* meIslBETMap_;
MonitorElement* meIslBCryMap_;

MonitorElement* meIslSEne_;
MonitorElement* meIslSNum_;
MonitorElement* meIslSSiz_;

MonitorElement* meIslSEneMap_;
MonitorElement* meIslSNumMap_;
MonitorElement* meIslSETMap_;
MonitorElement* meIslSSizMap_;

MonitorElement* meHybS1toE_;
MonitorElement* meInvMass_;

bool init_;

};

#endif
