#ifndef EECosmicTask_H
#define EECosmicTask_H

/*
 * \file EECosmicTask.h
 *
 * $Date: 2012/04/27 13:46:13 $
 * $Revision: 1.19 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

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
edm::InputTag EcalUncalibratedRecHitCollection_; 
edm::InputTag EcalRecHitCollection_;

MonitorElement* meSelMap_[18];

MonitorElement* meSpectrum_[2][18];

double threshold_;
double minJitter_;
double maxJitter_;

bool init_;

};

#endif
