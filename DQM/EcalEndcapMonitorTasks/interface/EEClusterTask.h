#ifndef EEClusterTask_H
#define EEClusterTask_H

/*
 * \file EEClusterTask.h
 *
 * $Date: 2007/10/18 08:22:42 $
 * $Revision: 1.6 $
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

edm::InputTag BasicClusterCollection_;
edm::InputTag SuperClusterCollection_;
 edm::InputTag ClusterShapeAssociation_;

MonitorElement* meBCEne_, *meBCNum_, *meBCSiz_;

MonitorElement* meBCEneFwdMap_, *meBCNumFwdMap_, *meBCETFwdMap_, *meBCSizFwdMap_;
MonitorElement* meBCEneFwdMapProjR_, *meBCNumFwdMapProjR_, *meBCETFwdMapProjR_, *meBCSizFwdMapProjR_;
MonitorElement* meBCEneFwdMapProjPhi_, *meBCNumFwdMapProjPhi_, *meBCETFwdMapProjPhi_, *meBCSizFwdMapProjPhi_;

MonitorElement* meBCEneBwdMap_, *meBCNumBwdMap_, *meBCETBwdMap_, *meBCSizBwdMap_;
MonitorElement* meBCEneBwdMapProjR_, *meBCNumBwdMapProjR_, *meBCETBwdMapProjR_, *meBCSizBwdMapProjR_;
MonitorElement* meBCEneBwdMapProjPhi_, *meBCNumBwdMapProjPhi_, *meBCETBwdMapProjPhi_, *meBCSizBwdMapProjPhi_;

MonitorElement* meSCEne_, *meSCNum_, *meSCSiz_;  

MonitorElement* mes1s9_, *mes9s25_, *meInvMass_;

bool init_;

};

#endif
