/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/08/03 16:10:19 $
 *  $Revision: 1.9 $
 *  \author C. Battilana - CIEMAT
 */


// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerSynchTest.h"

// Framework headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Geometry
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

// DB & Calib
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"


// Root
#include "TF1.h"
#include "TProfile.h"


//C++ headers
#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;


DTLocalTriggerSynchTest::DTLocalTriggerSynchTest(const edm::ParameterSet& ps) {

  setConfig(ps,"DTLocalTriggerSynch");
  baseFolderDCC = "DT/90-LocalTriggerSynch/";
  baseFolderDDU = "DT/90-LocalTriggerSynch/";

}


DTLocalTriggerSynchTest::~DTLocalTriggerSynchTest(){

}


void DTLocalTriggerSynchTest::beginJob(const edm::EventSetup& c){
  
  DTLocalTriggerBaseTest::beginJob(c);

  numHistoTag   = parameters.getParameter<string>("numHistoTag");
  denHistoTag   = parameters.getParameter<string>("denHistoTag");
  ratioHistoTag = parameters.getParameter<string>("ratioHistoTag");
  bxTime        = parameters.getParameter<double>("bxTimeInterval");   // CB move this to static const or DB
  rangeInBX     = parameters.getParameter<bool>("rangeWithinBX");

  vector<string>::const_iterator iTr   = trigSources.begin();
  vector<string>::const_iterator trEnd = trigSources.end();
  vector<string>::const_iterator iHw   = hwSources.begin();
  vector<string>::const_iterator hwEnd = hwSources.end();

  //Booking
  if(parameters.getUntrackedParameter<bool>("staticBooking", true)){
    for (; iTr != trEnd; ++iTr){
      trigSource = (*iTr);
      for (; iHw != hwEnd; ++iHw){
	hwSource = (*iHw);
	std::vector<DTChamber*>::const_iterator chambIt  = muonGeom->chambers().begin();
	std::vector<DTChamber*>::const_iterator chambEnd = muonGeom->chambers().end();
	for (; chambIt!=chambEnd; ++chambIt) { 
	  DTChamberId chId = ((*chambIt)->id());
	  bookChambHistos(chId,ratioHistoTag);
	}
      }
    }
  }
  
}


void DTLocalTriggerSynchTest::runClientDiagnostic() {

  // Loop over Trig & Hw sources
  for (vector<string>::const_iterator iTr = trigSources.begin(); iTr != trigSources.end(); ++iTr){
    trigSource = (*iTr);
    for (vector<string>::const_iterator iHw = hwSources.begin(); iHw != hwSources.end(); ++iHw){
      hwSource = (*iHw);
      std::vector<DTChamber*>::const_iterator chambIt  = muonGeom->chambers().begin();
      std::vector<DTChamber*>::const_iterator chambEnd = muonGeom->chambers().end();
      for (; chambIt!=chambEnd; ++chambIt) { 
	DTChamberId chId = (*chambIt)->id();
	uint32_t indexCh = chId.rawId();

	// Perform peak finding
	TH1F *numH     = getHisto<TH1F>(dbe->get(getMEName(numHistoTag,"", chId)));
	TH1F *denH     = getHisto<TH1F>(dbe->get(getMEName(denHistoTag,"", chId)));
	    
	if (numH && denH && numH->GetEntries()>1 && denH->GetEntries()>1) { // CB Set min entries from parameter	      
	  std::map<std::string,MonitorElement*> innerME = chambME[indexCh];
	  MonitorElement* ratioH = innerME.find(fullName(ratioHistoTag))->second;
	  makeRatioME(numH,denH,ratioH);
	  try {
	    getHisto<TH1F>(ratioH)->Fit("pol8","CQO");
	  } catch (...) {
	    edm::LogProblem(category()) << "[" << testName << "Test]: Error fitting " << ratioH->getName() << " returned 0" << endl;
	  }
	} else { 
	  if (!numH || !denH) {
	    LogProblem(category()) << "[" << testName << "Test]: At least one of the required Histograms was not found for chamber " << chId << " Peaks not computed" << endl;
	  } else {
	    LogProblem(category()) << "[" << testName << "Test]: Number of plots entries for " << chId << " is less than 1.  Peaks not computed" << endl;
	  }
	}

      }
    }
  }	

}

void DTLocalTriggerSynchTest::endJob(){

  DTLocalTriggerBaseTest::endJob();

  if ( parameters.getParameter<bool>("writeDB")) {
    LogProblem(category()) << "[" << testName << "Test]: writeDB flag set to true. Producing peak position database." << endl; // CB Fixe logger category	  

    DTTPGParameters* delayMap = new DTTPGParameters();
    hwSource =  parameters.getParameter<bool>("dbFromDCC") ? "DCC" : "DDU";
    std::vector<DTChamber*>::const_iterator chambIt  = muonGeom->chambers().begin();
    std::vector<DTChamber*>::const_iterator chambEnd = muonGeom->chambers().end();
      for (; chambIt!=chambEnd; ++chambIt) { 
	DTChamberId chId = (*chambIt)->id();
	float fineDelay  = 0;

	TH1F *ratioH     = getHisto<TH1F>(dbe->get(getMEName(ratioHistoTag,"", chId)));    
	if (ratioH->GetEntries()>1) { // CB Set min entries from parameter	      
	  TF1 *fitF=ratioH->GetFunction("pol8");
	  if (fitF) { fineDelay=fitF->GetMaximumX(0,bxTime); }
	}
	delayMap->set(chId,0,fineDelay,DTTimeUnits::ns);
      }

      string delayRecord = "DTTPGParametersRcd"; // CB Read from cfg???
      DTCalibDBUtils::writeToDB(delayRecord,delayMap);

      std::vector< std::pair<DTTPGParametersId,DTTPGParametersData> >::const_iterator dbIt  = delayMap->begin();
      std::vector< std::pair<DTTPGParametersId,DTTPGParametersData> >::const_iterator dbEnd = delayMap->end();
      for (; dbIt!=dbEnd; ++dbIt) {
	LogProblem(category()) << "[" << testName << "Test]: DB entry for Wh " << (*dbIt).first.wheelId 
			       << " Sec " << (*dbIt).first.sectorId 
			       << " St " << (*dbIt).first.wheelId 
			       << " has coarse " << (*dbIt).second.nClock
			       << " and phase " << (*dbIt).second.tPhase << std::endl;
      }
      
  }

}



void DTLocalTriggerSynchTest::makeRatioME(TH1F* numerator, TH1F* denominator, MonitorElement* result){
  
  TH1F* efficiency = result->getTH1F();
  efficiency->Divide(numerator,denominator,1,1,"");
  
}

// float DTLocalTriggerSynchTest::findMaximum(TH1F* histo) {

//   try {
//     histo->Fit("pol8","CQO");
//     TF1 *fitFunc= histo->GetFunction("pol8");
//     if (fitFunc) {
//       return fitFunc->GetMaximumX();
//     } else {
//       edm::LogProblem(category()) << "[" << testName << "Test]: Error getting maximum for " << histo->GetName() << " returned 0" << endl;
//       return 0;
//     }
//   } catch (...) {
//     edm::LogProblem(category()) << "[" << testName << "Test]: Error fitting " << histo->GetName() << " returned 0" << endl;
//   }

// }

void DTLocalTriggerSynchTest::bookChambHistos(DTChamberId chambId, string htype, string subfolder) {
  
  stringstream wheel; wheel << chambId.wheel();
  stringstream station; station << chambId.station();	
  stringstream sector; sector << chambId.sector();

  string fullType  = fullName(htype);
  bool isDCC = hwSource=="DCC" ;
  string HistoName = fullType + "_W" + wheel.str() + "_Sec" + sector.str() + "_St" + station.str();

  string folder = topFolder(isDCC) + "Wheel" + wheel.str() + "/Sector" + sector.str() + "/Station" + station.str();
  if ( subfolder!="") { folder += "7" + subfolder; }

  dbe->setCurrentFolder(folder);

  LogInfo(category()) << "[" << testName << "Test]: booking " << folder << "/" <<HistoName;

  
  uint32_t indexChId = chambId.rawId();
  float min = rangeInBX ?      0 :  -bxTime;
  float max = rangeInBX ? bxTime : 2*bxTime;
  int nbins = static_cast<int>(ceil( rangeInBX ? bxTime : 3*bxTime));

  chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"All/HH ratio vs Muon Arrival Time",nbins,min,max);

}
