/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/05/22 17:24:58 $
 *  $Revision: 1.5 $
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
#include "CondFormats/DataRecord/interface/DTTPGParametersRcd.h"
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


void DTLocalTriggerSynchTest::beginJob(){
  
  numHistoTag   = parameters.getParameter<string>("numHistoTag");
  denHistoTag   = parameters.getParameter<string>("denHistoTag");
  ratioHistoTag = parameters.getParameter<string>("ratioHistoTag");
  bxTime        = parameters.getParameter<double>("bxTimeInterval");
  rangeInBX     = parameters.getParameter<bool>("rangeWithinBX");
  nBXLow        = parameters.getParameter<int>("nBXLow");
  nBXHigh       = parameters.getParameter<int>("nBXHigh");
  minEntries    = parameters.getParameter<int>("minEntries");

}

void DTLocalTriggerSynchTest::beginRun(const Run& run, const EventSetup& c) {

  DTLocalTriggerBaseTest::beginRun(run,c);

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

  LogVerbatim(category()) << "[" << testName << "Test]: beginRun" << endl;

  if (parameters.getParameter<bool>("fineParamDiff")) {
    ESHandle<DTTPGParameters> wPhaseHandle;
    c.get<DTTPGParametersRcd>().get(wPhaseHandle);
    wPhaseMap = (*wPhaseHandle);
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
	    
	if (numH && denH && numH->GetEntries()>minEntries && denH->GetEntries()>minEntries) {	      
	  std::map<std::string,MonitorElement*> innerME = chambME[indexCh];
	  MonitorElement* ratioH = innerME.find(fullName(ratioHistoTag))->second;
	  makeRatioME(numH,denH,ratioH);
	  try {
	    getHisto<TH1F>(ratioH)->Fit("pol8","CQO");
	  } catch (cms::Exception& iException) {
	    edm::LogPrint(category()) << "[" << testName 
				     << "Test]: Error fitting " 
				     << ratioH->getName() << " returned 0" << endl;
	  }
	} else { 
	  if (!numH || !denH) {
	    LogPrint(category()) << "[" << testName 
				<< "Test]: At least one of the required Histograms was not found for chamber " 
				<< chId << ". Peaks not computed" << endl;
	  } else {
	    LogPrint(category()) << "[" << testName 
				<< "Test]: Number of plots entries for " 
				<< chId << " is less than minEntries=" 
				<< minEntries <<".  Peaks not computed" << endl;
	  }
	}

      }
    }
  }	

}

void DTLocalTriggerSynchTest::endJob(){

  DTLocalTriggerBaseTest::endJob();

  if ( parameters.getParameter<bool>("writeDB")) {
    LogVerbatim(category()) << "[" << testName 
			    << "Test]: writeDB flag set to true. Producing peak position database." << endl;

    DTTPGParameters* delayMap = new DTTPGParameters();
    hwSource =  parameters.getParameter<bool>("dbFromDCC") ? "DCC" : "DDU";
    std::vector<DTChamber*>::const_iterator chambIt  = muonGeom->chambers().begin();
    std::vector<DTChamber*>::const_iterator chambEnd = muonGeom->chambers().end();
      for (; chambIt!=chambEnd; ++chambIt) { 

	DTChamberId chId = (*chambIt)->id();
	float fineDelay = 0;
	int coarseDelay = static_cast<int>((getFloatFromME(chId,"tTrig_SL1") + getFloatFromME(chId,"tTrig_SL3"))*0.5/bxTime);

	bool fineDiff   = parameters.getParameter<bool>("fineParamDiff");
	bool coarseDiff = parameters.getParameter<bool>("coarseParamDiff");


	TH1F *ratioH     = getHisto<TH1F>(dbe->get(getMEName(ratioHistoTag,"", chId)));    
	if (ratioH->GetEntries()>minEntries) {	      
	  TF1 *fitF=ratioH->GetFunction("pol8");
	  if (fitF) { fineDelay=fitF->GetMaximumX(0,bxTime); }
	} else {
	  LogInfo(category()) << "[" << testName 
			      << "Test]: Ratio histogram for chamber " << chId
			      << " is empty. Worst Phase value set to 0." << endl;
	}

	if (fineDiff || coarseDiff) {
	  float wFine;
	  int wCoarse;
	  wPhaseMap.get(chId,wCoarse,wFine,DTTimeUnits::ns);
	  if (fineDiff)   { fineDelay = wFine - fineDelay; }
	  if (coarseDiff) { coarseDelay = wCoarse - coarseDelay; }
	} 
	delayMap->set(chId,coarseDelay,fineDelay,DTTimeUnits::ns);
      }

      std::vector< std::pair<DTTPGParametersId,DTTPGParametersData> >::const_iterator dbIt  = delayMap->begin();
      std::vector< std::pair<DTTPGParametersId,DTTPGParametersData> >::const_iterator dbEnd = delayMap->end();
      for (; dbIt!=dbEnd; ++dbIt) {
	LogVerbatim(category()) << "[" << testName << "Test]: DB entry for Wh " << (*dbIt).first.wheelId 
				<< " Sec " << (*dbIt).first.sectorId 
				<< " St " << (*dbIt).first.stationId 
				<< " has coarse " << (*dbIt).second.nClock
				<< " and phase " << (*dbIt).second.tPhase << std::endl;
      }
      
      string delayRecord = "DTTPGParametersRcd";
      DTCalibDBUtils::writeToDB(delayRecord,delayMap);
      
  }

}



void DTLocalTriggerSynchTest::makeRatioME(TH1F* numerator, TH1F* denominator, MonitorElement* result){
  
  TH1F* efficiency = result->getTH1F();
  efficiency->Divide(numerator,denominator,1,1,"");
  
}

float DTLocalTriggerSynchTest::getFloatFromME(DTChamberId chId, std::string meType) {
   
   stringstream wheel; wheel << chId.wheel();
   stringstream station; station << chId.station();
   stringstream sector; sector << chId.sector();

   string folderName = topFolder(hwSource=="DCC") + "Wheel" +  wheel.str() +
     "/Sector" + sector.str() + "/Station" + station.str() + "/" ; 

   string histoname = sourceFolder + folderName 
     + meType
     + "_W" + wheel.str()  
     + "_Sec" + sector.str()
     + "_St" + station.str();

   MonitorElement* me = dbe->get(histoname);
   if (me) { 
     return me->getFloatValue(); 
   }
   else { 
     LogProblem(category()) << "[" << testName << "Test]: " << histoname << " is not a valid ME. 0 returned" << std::endl;
   }
   
   return 0;

 }

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

  LogPrint(category()) << "[" << testName << "Test]: booking " << folder << "/" <<HistoName;

  
  uint32_t indexChId = chambId.rawId();
  float min = rangeInBX ?      0 : nBXLow*bxTime;
  float max = rangeInBX ? bxTime : nBXHigh*bxTime;
  int nbins = static_cast<int>(ceil( rangeInBX ? bxTime : (nBXHigh-nBXLow)*bxTime));

  chambME[indexChId][fullType] = dbe->book1D(HistoName.c_str(),"All/HH ratio vs Muon Arrival Time",nbins,min,max);

}
