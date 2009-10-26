#ifndef EEClusterTask_H
#define EEClusterTask_H

/*
 * \file EEClusterTask.h
 *
 * $Date: 2009/02/26 14:54:45 $
 * $Revision: 1.17 $
 * \author G. Della Ricca
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

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
void beginJob(void);

/// EndJob
void endJob(void);

/// BeginRun
void beginRun(const edm::Run & r, const edm::EventSetup & c);

/// EndRun
void endRun(const edm::Run & r, const edm::EventSetup & c);

/// Reset
void reset(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

edm::InputTag EcalRawDataCollection_;
edm::InputTag BasicClusterCollection_;
edm::InputTag SuperClusterCollection_;
edm::InputTag EcalRecHitCollection_;

MonitorElement* meBCEne_, *meBCNum_, *meBCSiz_;

MonitorElement* meBCEneFwdMap_, *meBCNumFwdMap_, *meBCETFwdMap_, *meBCSizFwdMap_;
MonitorElement* meBCEneFwdMapProjR_, *meBCNumFwdMapProjR_, *meBCETFwdMapProjR_, *meBCSizFwdMapProjR_;
MonitorElement* meBCEneFwdMapProjPhi_, *meBCNumFwdMapProjPhi_, *meBCETFwdMapProjPhi_, *meBCSizFwdMapProjPhi_;

MonitorElement* meBCEneBwdMap_, *meBCNumBwdMap_, *meBCETBwdMap_, *meBCSizBwdMap_;
MonitorElement* meBCEneBwdMapProjR_, *meBCNumBwdMapProjR_, *meBCETBwdMapProjR_, *meBCSizBwdMapProjR_;
MonitorElement* meBCEneBwdMapProjPhi_, *meBCNumBwdMapProjPhi_, *meBCETBwdMapProjPhi_, *meBCSizBwdMapProjPhi_;

MonitorElement* meSCEne_, *meSCNum_, *meSCSiz_;  

MonitorElement* meSCCrystalSiz_;
MonitorElement* meSCSeedEne_;
MonitorElement* meSCEne2_;
MonitorElement* meSCEneVsEMax_;
MonitorElement* meSCEneLowScale_;
MonitorElement* meSCSeedMapOcc_[2];
MonitorElement* meSCMapSingleCrystal_[2];
MonitorElement* meSCSeedTimingSummary_;
MonitorElement* meSCSeedTimingMap_[2];
MonitorElement* meSCSeedTiming_[18];

MonitorElement* mes1s9_, *mes9s25_;
MonitorElement* meInvMassPi0_;
MonitorElement* meInvMassJPsi_;
MonitorElement* meInvMassZ0_;
MonitorElement* meInvMassHigh_;

bool init_;

};

#endif
