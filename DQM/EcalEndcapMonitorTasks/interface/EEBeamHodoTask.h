#ifndef EEBeamHodoTask_H
#define EEBeamHodoTask_H

/*
 * \file EEBeamHodoTask.h
 *
 * $Date: 2008/04/08 15:06:27 $
 * $Revision: 1.8 $
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
void beginJob(const edm::EventSetup& c);

/// EndJob
void endJob(void);

/// Setup
void setup(void);

/// Cleanup
void cleanup(void);

private:

int ievt_;

DQMStore* dqmStore_;

std::string prefixME_;

bool enableCleanup_;

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
