#ifndef EEClusterTaskExtras_H
#define EEClusterTaskExtras_H

/*
 * \file EEClusterTaskExtras.h
 *
 * $Date: 2009/12/14 21:14:06 $
 * $Revision: 1.5 $
 * \author G. Della Ricca
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#define EECLUSTERTASKEXTRAS_DQMOFFLINE

class MonitorElement;
class DQMStore;

class EEClusterTaskExtras: public edm::EDAnalyzer{

 public:

/// Constructor
EEClusterTaskExtras(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEClusterTaskExtras();

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

// Determine Trigger
std::vector<bool> determineTriggers(const edm::Event&, const edm::EventSetup& eventSetup);
bool isExclusiveTrigger(int l1Trigger, std::vector<bool>& l1Triggers);
bool doMonitorElement(std::string meName);

enum L1Triggers {
   CSC_TRIGGER	= 0,
   DT_TRIGGER	= 1,
   ECAL_TRIGGER	= 2,
   HCAL_TRIGGER	= 3,
   RPC_TRIGGER	= 4
};

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

bool mergeRuns_;

edm::InputTag SuperClusterCollection_;
edm::InputTag EcalRecHitCollection_;
edm::InputTag l1GMTReadoutRecTag_;
edm::InputTag l1GTReadoutRecTag_;
std::vector<std::string> meList_;

#ifndef EECLUSTERTASKEXTRAS_DQMOFFLINE
MonitorElement* meSCSizCrystal_;
MonitorElement* meSCSizBC_;

MonitorElement* meSCSeedEne_;
MonitorElement* meSCEne2_;
MonitorElement* meSCEneLow_;
MonitorElement* meSCEneHigh_;
MonitorElement* meSCEneSingleCrystal_;

MonitorElement* meSCSeedMapOccSC_[2];
MonitorElement* meSCSeedMapOccHighEne_[2];
MonitorElement* meSCSeedMapOccSingleCrystal_[2];
MonitorElement* meSCSeedMapOccTrg_[2][5];
MonitorElement* meSCSeedMapOccTrgExcl_[2][5];
MonitorElement* meSCSeedTime_;
MonitorElement* meSCSeedMapTimeSC_[2];
MonitorElement* meSCSeedTimeVsAmp_;
MonitorElement* meSCSeedTimeEEM_;
MonitorElement* meSCSeedTimeEEP_;
MonitorElement* meSCSeedTimePerFed_[18];
MonitorElement* meSCSeedMapTimeSC_[2][5];
#endif

MonitorElement* meSCSizCrystalVsEne_;

MonitorElement* meSCSeedMapOcc_[2];
MonitorElement* meSCSeedMapOccHighEneSC_[2];
MonitorElement* meSCSeedMapOccTrgSC_[2][5];
MonitorElement* meSCSeedMapOccTrgExclSC_[2][5];

MonitorElement* meSCSeedMapTimeTrgSC_[2][5];

bool init_;

};

#endif
