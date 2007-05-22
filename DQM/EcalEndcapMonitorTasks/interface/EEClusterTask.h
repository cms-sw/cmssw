#ifndef EEClusterTask_H
#define EEClusterTask_H

/*
 * \file EEClusterTask.h
 *
 * $Date: 2007/04/05 14:54:03 $
 * $Revision: 1.3 $
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

MonitorElement* meEne_;
MonitorElement* meNum_;
MonitorElement* meSiz_;

MonitorElement* meEneFwdMap_;
MonitorElement* meNumFwdMap_;
MonitorElement* meEneFwdPolarMap_;
MonitorElement* meNumFwdPolarMap_;
MonitorElement* meEneBwdMap_;
MonitorElement* meNumBwdMap_;
MonitorElement* meEneBwdPolarMap_;
MonitorElement* meNumBwdPolarMap_;

MonitorElement* meInvMass_;

bool init_;

};

#endif
