#include "DQMOffline/CalibMuon/interface/DTtTrigDBValidation.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// DataFormats
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

// tTrig
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include <stdio.h>
#include <sstream>
#include <math.h>
#include "TFile.h"

using namespace edm;
using namespace std;

DTtTrigDBValidation::DTtTrigDBValidation(const ParameterSet& pset):
   metname_("TTrigDBValidation"),
   labelDBRef_(pset.getParameter<string>("labelDBRef")),
   labelDB_(pset.getParameter<string>("labelDB")),
   lowerLimit_(pset.getUntrackedParameter<int>("lowerLimit",1)),
   higherLimit_(pset.getUntrackedParameter<int>("higherLimit",3))
 {

  LogVerbatim(metname_) << "[DTtTrigDBValidation] Constructor called!";
}

DTtTrigDBValidation::~DTtTrigDBValidation(){}

void DTtTrigDBValidation::bookHistograms(DQMStore::IBooker &iBooker,
  edm::Run const &, edm::EventSetup const &setup) {

  LogVerbatim(metname_) << "[DTtTrigDBValidation] Parameters initialization";
  iBooker.setCurrentFolder("DT/DtCalib/TTrigDBValidation");
 
  ESHandle<DTTtrig> tTrig_Ref;
  setup.get<DTTtrigRcd>().get(labelDBRef_, tTrig_Ref);
  const DTTtrig* DTTtrigRefMap = &*tTrig_Ref;
  LogVerbatim(metname_) << "[DTtTrigDBValidation] reference Ttrig version: " << tTrig_Ref->version();

  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(labelDB_, tTrig);
  const DTTtrig* DTTtrigMap = &*tTrig;
  LogVerbatim(metname_) << "[DTtTrigDBValidation] Ttrig to validate version: " << tTrig->version();

  //book&reset the summary histos
  for(int wheel=-2; wheel<=2; wheel++){
    bookHistos(iBooker, wheel);
    tTrigDiffWheel_[wheel]->Reset();
  }

  // Get the geometry
  setup.get<MuonGeometryRecord>().get(dtGeom_);

  // Loop over Ref DB entries
  for(DTTtrig::const_iterator it = DTTtrigRefMap->begin();
      it != DTTtrigRefMap->end(); ++it) {
    DTSuperLayerId slId((*it).first.wheelId,
		        (*it).first.stationId,
		        (*it).first.sectorId,
		        (*it).first.slId);
    float tTrigMean;
    float tTrigRms;
    float kFactor;
    DTTtrigRefMap->get(slId, tTrigMean, tTrigRms, kFactor, DTTimeUnits::ns);
    float tTrigCorr = tTrigMean + kFactor*tTrigRms;
    LogTrace(metname_)<< "Ref Superlayer: " <<  slId << "\n"
		     << " Ttrig mean (ns): " << tTrigMean
		     << " Ttrig rms (ns): " << tTrigRms
                     << " Ttrig k-Factor: " << kFactor
                     << " Ttrig value (ns): " << tTrigCorr;
                     
    //tTrigRefMap[slId] = std::pair<float,float>(tTrigmean,tTrigrms);
    tTrigRefMap_[slId] = pair<float,float>(tTrigCorr,tTrigRms);
  }

  // Loop over Ref DB entries
  for(DTTtrig::const_iterator it = DTTtrigMap->begin();
      it != DTTtrigMap->end(); ++it) {
    DTSuperLayerId slId((*it).first.wheelId,
		        (*it).first.stationId,
		        (*it).first.sectorId,
		        (*it).first.slId);
    float tTrigMean;
    float tTrigRms;
    float kFactor;
    DTTtrigMap->get(slId, tTrigMean, tTrigRms, kFactor, DTTimeUnits::ns);
    float tTrigCorr = tTrigMean + kFactor*tTrigRms;
    LogTrace(metname_)<< "Superlayer: " <<  slId << "\n"
                     << " Ttrig mean (ns): " << tTrigMean
                     << " Ttrig rms (ns): " << tTrigRms
                     << " Ttrig k-Factor: " << kFactor
                     << " Ttrig value (ns): " << tTrigCorr;

    //tTrigMap[slId] = std::pair<float,float>(tTrigmean,tTrigrms);
    tTrigMap_[slId] = pair<float,float>(tTrigCorr,tTrigRms);
  }

  for(map<DTSuperLayerId, pair<float,float> >::const_iterator it = tTrigRefMap_.begin();
      it != tTrigRefMap_.end(); ++it) {  
      if(tTrigMap_.find((*it).first) == tTrigMap_.end()) continue;

      // compute the difference
      float difference = tTrigMap_[(*it).first].first - (*it).second.first;

      //book histo
      int wheel = (*it).first.chamberId().wheel();
      int sector = (*it).first.chamberId().sector();	
      if(tTrigDiffHistos_.find(make_pair(wheel,sector)) == tTrigDiffHistos_.end()) bookHistos(iBooker, wheel, sector);
			
      LogTrace(metname_) << "Filling histos for super-layer: " << (*it).first << "  difference: " << difference;
 
      // Fill the test histos
      int entry = -1;
      int station = (*it).first.chamberId().station();	
      if(station == 1) entry=0;
      if(station == 2) entry=3;
      if(station == 3) entry=6;
      if(station == 4) entry=9;

      int slBin = entry + (*it).first.superLayer();
      if(slBin == 12) slBin=11;	

      tTrigDiffHistos_[make_pair(wheel,sector)]->setBinContent(slBin, difference);
      if(abs(difference) < lowerLimit_){
	tTrigDiffWheel_[wheel]->setBinContent(slBin,sector,1);
      }else if(abs(difference) < higherLimit_){
	tTrigDiffWheel_[wheel]->setBinContent(slBin,sector,2);
      }else{
	tTrigDiffWheel_[wheel]->setBinContent(slBin,sector,3);
      }
	

  } // Loop over the tTrig map reference
  
}

void DTtTrigDBValidation::bookHistos(DQMStore::IBooker &iBooker, int wheel, int sector) {

  LogTrace(metname_) << "   Booking histos for Wheel, Sector: " << wheel << ", " << sector;

  // Compose the chamber name
  stringstream str_wheel; str_wheel << wheel;
  stringstream str_sector; str_sector << sector;

  string lHistoName = "_W" + str_wheel.str() + "_Sec" + str_sector.str();

  iBooker.setCurrentFolder("DT/DtCalib/TTrigDBValidation/Wheel" + str_wheel.str());

  // Create the monitor elements
  MonitorElement * hDifference;
  hDifference = iBooker.book1D("TTrigDifference"+lHistoName, "difference between the two tTrig values",11,0,11);

  pair<int,int> mypair(wheel,sector);
  tTrigDiffHistos_[mypair] = hDifference;

  (tTrigDiffHistos_[mypair])->setBinLabel(1,"MB1_SL1",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(2,"MB1_SL2",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(3,"MB1_SL3",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(4,"MB2_SL1",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(5,"MB2_SL2",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(6,"MB2_SL3",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(7,"MB3_SL1",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(8,"MB3_SL2",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(9,"MB3_SL3",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(10,"MB4_SL1",1);
  (tTrigDiffHistos_[mypair])->setBinLabel(11,"MB4_SL3",1);

}

// Book the summary histos
void DTtTrigDBValidation::bookHistos(DQMStore::IBooker &iBooker, int wheel) {

  stringstream wh; wh << wheel;

  iBooker.setCurrentFolder("DT/DtCalib/TTrigDBValidation");
  tTrigDiffWheel_[wheel] = iBooker.book2D("TTrigDifference_W"+wh.str(), "W"+wh.str()+": summary of tTrig differences",11,1,12,14,1,15);
  tTrigDiffWheel_[wheel]->setBinLabel(1,"MB1_SL1",1);
  tTrigDiffWheel_[wheel]->setBinLabel(2,"MB1_SL2",1);
  tTrigDiffWheel_[wheel]->setBinLabel(3,"MB1_SL3",1);
  tTrigDiffWheel_[wheel]->setBinLabel(4,"MB2_SL1",1);
  tTrigDiffWheel_[wheel]->setBinLabel(5,"MB2_SL2",1);
  tTrigDiffWheel_[wheel]->setBinLabel(6,"MB2_SL3",1);
  tTrigDiffWheel_[wheel]->setBinLabel(7,"MB3_SL1",1);
  tTrigDiffWheel_[wheel]->setBinLabel(8,"MB3_SL2",1);
  tTrigDiffWheel_[wheel]->setBinLabel(9,"MB3_SL3",1);
  tTrigDiffWheel_[wheel]->setBinLabel(10,"MB4_SL1",1);
  tTrigDiffWheel_[wheel]->setBinLabel(11,"MB4_SL3",1);

}


int DTtTrigDBValidation::stationFromBin(int bin) const {
  return (int) (bin /3.1)+1;
}
 
int DTtTrigDBValidation::slFromBin(int bin) const {
  int ret = bin%3;
  if(ret == 0 || bin == 11) ret = 3;
  
  return ret;
}


void DTtTrigDBValidation::analyze( const edm::Event&, const edm::EventSetup&) {}
