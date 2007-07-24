#ifndef EEClusterTask_H
#define EEClusterTask_H

/*
 * \file EEClusterTask.h
 *
 * $Date: 2007/05/22 15:08:12 $
 * $Revision: 1.4 $
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

edm::InputTag islandEndcapSuperClusterCollection_;
edm::InputTag islandEndcapBasicClusterCollection_;

MonitorElement* meEne_, *meEneBasic_;
MonitorElement* meNum_, *meNumBasic_;
MonitorElement* meSiz_, *meSizBasic_;

MonitorElement* meEneFwdMap_, *meEneFwdMapBasic_;
MonitorElement* meNumFwdMap_, *meNumFwdMapBasic_;
MonitorElement* meEneFwdPolarMap_, *meEneFwdPolarMapBasic_;
MonitorElement* meNumFwdPolarMap_, *meNumFwdPolarMapBasic_;
MonitorElement* meEneBwdMap_, *meEneBwdMapBasic_;
MonitorElement* meNumBwdMap_, *meNumBwdMapBasic_;
MonitorElement* meEneBwdPolarMap_, *meEneBwdPolarMapBasic_;
MonitorElement* meNumBwdPolarMap_, *meNumBwdPolarMapBasic_;

MonitorElement* meInvMass_;

bool init_;

};

#endif
