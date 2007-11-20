#ifndef EBClusterTask_H
#define EBClusterTask_H

/*
 * \file EBClusterTask.h
 *
 * $Date: 2007/10/23 07:11:58 $
 * $Revision: 1.12 $
 * \author G. Della Ricca
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DaqMonitorBEInterface;

class EBClusterTask: public edm::EDAnalyzer{

 public:

/// Constructor
EBClusterTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBClusterTask();

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

MonitorElement* meBCEne_;
MonitorElement* meBCNum_;
MonitorElement* meBCSiz_;

MonitorElement* meBCEneMap_, *meBCEneMapProjEta_, *meBCEneMapProjPhi_;
MonitorElement* meBCNumMap_, *meBCNumMapProjEta_, *meBCNumMapProjPhi_;
MonitorElement* meBCETMap_, *meBCETMapProjEta_, *meBCETMapProjPhi_;
MonitorElement* meBCSizMap_, *meBCSizMapProjEta_, *meBCSizMapProjPhi_;

MonitorElement* meSCEne_;
MonitorElement* meSCNum_;
MonitorElement* meSCSiz_;

MonitorElement* mes1s9_;
MonitorElement* mes9s25_;
MonitorElement* meInvMass_;

bool init_;

};

#endif
