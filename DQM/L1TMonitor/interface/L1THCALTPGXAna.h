#ifndef L1THCALTPGXANA_H
#define L1THCALTPGXANA_H

/*
 * \file L1THCALTPGXAna.h
 *
 * $Date: 2007/02/23 22:00:16 $
 * $Revision: 1.3 $
 * \author J. Berryhill
 *
*/

// system include files
#include <memory>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//adam's includes
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"
//#include "DQM/L1TMonitor/interface/hcal_root_prefs.h"


//
// class decleration
//

class L1THCALTPGXAna : public edm::EDAnalyzer {

public:

  typedef std::multimap<HcalTrigTowerDetId, double> IdtoEnergy;


// Constructor
L1THCALTPGXAna(const edm::ParameterSet& ps);

// Destructor
virtual ~L1THCALTPGXAna();

protected:
// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);
 
 private:
// ----------member data ---------------------------
 DaqMonitorBEInterface * dbe;
 HcalTrigTowerGeometry theTrigTowerGeometry;
 IdtoEnergy Rec_towers;
 
 // what we monitor
 MonitorElement *hcalTpEtEtaPhi_;
 MonitorElement *hcalTpOccEtaPhi_;
 MonitorElement *hcalTpRank_;
 MonitorElement *hcalEffDen_1_;
 MonitorElement *hcalEffNum_1_;
 MonitorElement *hcalEffDen_2_;
 MonitorElement *hcalEffNum_2_;
 MonitorElement *hcalEffDen_3_;
 MonitorElement *hcalEffNum_3_;
 MonitorElement *hcalEffDen_4_;
 MonitorElement *hcalEffNum_4_;
 MonitorElement *hcalFakes_;
 MonitorElement *hcalNoFire_;
 MonitorElement *hcalTpgRatiom1_;
 MonitorElement *hcalTpgRatioSOI_;
 MonitorElement *hcalTpgRatiop1_;
 MonitorElement *hcalTpgRatiop2_;
 MonitorElement *hcalTpgvsRec1_;
 MonitorElement *hcalTpgvsRec2_;
 MonitorElement *hcalTpgvsRec3_;
 MonitorElement *hcalTpgvsRec4_;
 MonitorElement *hcalTpSat_;
 MonitorElement *hcalEffNum_HBHE[56][72]; //56 eta slices, 72 phi slices
 MonitorElement *hcalEffNum_HF[8][18]; //8 eta slices, 18 phi slices
 MonitorElement *hcalEffDen_HBHE[56][72]; //56 eta slices, 72 phi slices
 MonitorElement *hcalEffDen_HF[8][18]; //8 eta slices, 18 phi slices
 MonitorElement *hcalTpgfgperbunch_;
 MonitorElement *hcalTpgfgbindiff_;
 MonitorElement *hcalTpgfgtimediff_;

 int nev_; // Number of events processed
 std::string outputFile_; //file name for ROOT ouput
 bool verbose_;
 bool monitorDaemon_;
 ofstream logFile_;
 double fakeCut_;
 edm::InputTag hcaltpgSource_;
 edm::InputTag hbherecoSource_;
 edm::InputTag hfrecoSource_;
 int numFG;
 int binfg1;
 int binfg2;

};

double find_eta(double,double);

#endif
