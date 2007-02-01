#ifndef EBBeamCaloTask_H
#define EBBeamCaloTask_H

/*
 * \file EBBeamCaloTask.h
 *
 * $Date: 2006/08/09 15:03:56 $
 * $Revision: 1.16 $
 * \author A. Ghezzi
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "TBDataFormats/EcalTBObjects/interface/EcalTBCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <vector>

class EBBeamCaloTask: public edm::EDAnalyzer{

public:

/// Constructor
EBBeamCaloTask(const edm::ParameterSet& ps);

/// Destructor
virtual ~EBBeamCaloTask();

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
 const static int cryInArray_ = 9;
 const static int defaultPede_ = 200;
int ievt_;

std::string digiProducer_;
//std::string DCCHeaderProducer_;

MonitorElement* meBBCaloPulseProf_[cryInArray_];
MonitorElement* meBBCaloPulseProfG12_[cryInArray_];
MonitorElement* meBBCaloGains_[cryInArray_];
MonitorElement* meBBCaloEne_[cryInArray_];

//MonitorElement* meBBCaloPulseProfMoving_[cryInArray_];
//MonitorElement* meBBCaloPulseProfG12Moving_[cryInArray_];
//MonitorElement* meBBCaloGainsMoving_[cryInArray_];
//MonitorElement* meBBCaloEneMoving_[cryInArray_];

MonitorElement* meBBCaloCryRead_;
//MonitorElement* meBBCaloCryReadMoving_;

MonitorElement* meBBCaloAllNeededCry_;
MonitorElement* meBBNumCaloCryRead_;

MonitorElement* meBBCaloE3x3_;
MonitorElement* meBBCaloE3x3Moving_;

/* MonitorElement* meBBCaloE3x3Cry_[1701]; */
/* MonitorElement* meBBCaloE1Cry_[1701]; */

MonitorElement* meBBCaloCryOnBeam_;
MonitorElement* meBBCaloMaxEneCry_;

MonitorElement* TableMoving_;

MonitorElement* CrystalsDone_;

MonitorElement* CrystalInBeam_vs_Event_;

MonitorElement* meEBBCaloReadCryErrors_;

MonitorElement* meEBBCaloE1vsCry_;

MonitorElement* meEBBCaloE3x3vsCry_;

MonitorElement* meEBBCaloEntriesVsCry_;

MonitorElement* meEBBCaloBeamCentered_;

MonitorElement* meEBBCaloE1MaxCry_;

MonitorElement* meEBBCaloDesync_;

bool init_; 

bool profileArranged_;

int PreviousTableStatus_[2];
int PreviousCrystalinBeam_[3];

 int cib_[12];// used 10
bool changed_tb_status_;
bool changed_cry_in_beam_;
int evt_after_change_ ;
bool  wasFakeChange_;
int lastStableStatus_ ;
int table_step_, crystal_step_;
int event_last_reset_;
int last_cry_in_beam_;
int previous_cry_in_beam_;
int previous_ev_num_; 
// 0=stable, 1=moving, 
// PreviousTableStatus_[0]-> event=current -2
// PreviousTableStatus_[1]-> event=current -1

//int cryIn3x3_[cryInArray_];

};

#endif
