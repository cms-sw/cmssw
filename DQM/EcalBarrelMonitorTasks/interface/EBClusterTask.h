#ifndef EBClusterTask_H
#define EBClusterTask_H

/*
 * \file EBClusterTask.h
 *
 * \author G. Della Ricca
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

class MonitorElement;
class DQMStore;

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

edm::EDGetTokenT<EcalRawDataCollection> EcalRawDataCollection_;
edm::EDGetTokenT<reco::BasicClusterCollection> BasicClusterCollection_;
edm::EDGetTokenT<reco::SuperClusterCollection> SuperClusterCollection_;
edm::EDGetTokenT<EcalRecHitCollection> EcalRecHitCollection_;

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

MonitorElement* meSCCrystalSiz_;
MonitorElement* meSCSeedEne_;
MonitorElement* meSCEne2_;
MonitorElement* meSCEneVsEMax_;
MonitorElement* meSCEneLowScale_;
MonitorElement* meSCSeedMapOcc_;
MonitorElement* meSCMapSingleCrystal_;

MonitorElement* mes1s9_;
MonitorElement* mes1s9thr_;
MonitorElement* mes9s25_;
MonitorElement* meInvMassPi0_;
MonitorElement* meInvMassJPsi_;
MonitorElement* meInvMassZ0_;
MonitorElement* meInvMassHigh_;
MonitorElement* meInvMassPi0Sel_;
MonitorElement* meInvMassJPsiSel_;
MonitorElement* meInvMassZ0Sel_;
MonitorElement* meInvMassHighSel_;

bool init_;

float thrS4S9_, thrClusEt_, thrCandEt_;

};

#endif
