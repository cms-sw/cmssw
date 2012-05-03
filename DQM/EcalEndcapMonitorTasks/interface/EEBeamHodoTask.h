#ifndef EEBeamHodoTask_H
#define EEBeamHodoTask_H

/*
 * \file EEBeamHodoTask.h
 *
 * $Date: 2008/05/11 09:35:11 $
 * $Revision: 1.10 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class MonitorElement;
class DQMStore;

class EEBeamHodoTask: public edm::EDAnalyzer{

public:

/// Constructor
EEBeamHodoTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EEBeamHodoTask();

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

edm::InputTag EcalTBEventHeader_;
edm::InputTag EcalRawDataCollection_;
edm::InputTag EcalUncalibratedRecHitCollection_;
edm::InputTag EcalTBTDCRawInfo_;
edm::InputTag EcalTBHodoscopeRawInfo_;
edm::InputTag EcalTBTDCRecInfo_;
edm::InputTag EcalTBHodoscopeRecInfo_;

int LV1_;
bool tableIsMoving_;
bool resetNow_;
int    cryInBeam_;
int    previousCryInBeam_;
int    cryInBeamCounter_;

 //  ME type I
MonitorElement* meHodoOcc_[4];
MonitorElement* meHodoRaw_[4];
MonitorElement* meHodoPosRecXY_;
MonitorElement* meHodoPosRecX_;
MonitorElement* meHodoPosRecY_;
MonitorElement* meHodoSloXRec_;
MonitorElement* meHodoSloYRec_;
MonitorElement* meHodoQuaXRec_;
MonitorElement* meHodoQuaYRec_;
MonitorElement* meTDCRec_;
MonitorElement* meHodoPosXMinusCaloPosXVsCry_;
MonitorElement* meHodoPosYMinusCaloPosYVsCry_;
MonitorElement* meTDCTimeMinusCaloTimeVsCry_;
MonitorElement* meMissingCollections_;

 //  ME type II
MonitorElement* meEvsXRecProf_;
MonitorElement* meEvsYRecProf_;
MonitorElement* meEvsXRecHis_;
MonitorElement* meEvsYRecHis_;
MonitorElement* meCaloVsHodoXPos_;
MonitorElement* meCaloVsHodoYPos_;
MonitorElement* meCaloVsTDCTime_;

bool init_;

int smId;

};

#endif
