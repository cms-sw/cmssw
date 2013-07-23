/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/11/24 09:17:30 $
 *  $Revision: 1.5 $
 *  \author M. Giunta, C. Battilana 
 */


// This class header
#include "DQM/DTMonitorClient/src/DTFineDelayCorr.h"

// Framework headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

// L1Trigger
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigTrigUnit.h"

// Geometry
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

// DB & Calib
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

// Root
#include "TF1.h"
#include "TProfile.h"


//C++ headers
using namespace edm;
using namespace std;
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <vector>
#include <map>

DTFineDelayCorr::DTFineDelayCorr(const ParameterSet& ps) {

  setConfig(ps,"DTFineDelayCorr");  // sets parameter values and name used in log file 
  baseFolderDCC = "DT/90-LocalTriggerSynch/";
  baseFolderDDU = "DT/90-LocalTriggerSynch/";
 
}


DTFineDelayCorr::~DTFineDelayCorr(){

}


void DTFineDelayCorr::beginJob(){

  // Tag for Hardware Source (DDU or DCC)
  hwSource = parameters.getParameter<string>("hwSource");
  // Tag for the t0Mean Histograms
  t0MeanHistoTag = parameters.getParameter<string>("t0MeanHistoTag");
  // Read old delays from file or from Db
  readOldFromDb = parameters.getParameter<bool>("readOldFromDb");
  // Input file name for old delays
  oldDelaysInputFile = parameters.getParameter<string>("oldDelaysInputFile"),
  // Write new delays to file or to Db
  writeDB = parameters.getParameter<bool>("writeDB");
  // Output File Name
  outputFileName = parameters.getParameter<string>("outputFile");
  // Choose to use Hist Mean or Gaussian Fit Mean
  gaussMean =  parameters.getParameter<bool>("gaussMean");
  // Require Minimum Number Of Entries in the t0Mean Histogram
  minEntries =  parameters.getUntrackedParameter<int>("minEntries",5);

}

void DTFineDelayCorr::beginRun(const Run& run, const EventSetup& evSU){

  DTLocalTriggerBaseTest::beginRun(run,evSU);
  evSU.get< DTConfigManagerRcd >().get(dtConfig);
  evSU.get< DTTPGParametersRcd >().get(worstPhaseMap);

}

void DTFineDelayCorr::runClientDiagnostic() {
  int coarseDelay = -999;
  float oldFineDelay = -999;
  if(!readOldFromDb) { // read old delays from txt file
    // **  Open and read old delays input file  ** 
    ifstream oldDelaysFile(oldDelaysInputFile.c_str());
    string line;

    while (getline(oldDelaysFile, line)) {
      if( line == "" || line[0] == '#' ) continue; 
      stringstream linestr;
      int wheelKey,sectorKey, stationKey;      
      linestr << line;
      
      linestr >> wheelKey
	      >> sectorKey
	      >> stationKey
	      >> coarseDelay
	      >> oldFineDelay;
      
      pair<int,float> oldDelays = make_pair(coarseDelay,oldFineDelay);
      DTChamberId oldDelayKey = DTChamberId(wheelKey,stationKey,sectorKey);
      oldDelayMap.insert(make_pair(oldDelayKey,oldDelays));
    }
  }

  //  ** Loop over the chambers ** 
  vector<DTChamber*>::const_iterator chambIt  = muonGeom->chambers().begin();
  vector<DTChamber*>::const_iterator chambEnd = muonGeom->chambers().end();
  for (; chambIt!=chambEnd; ++chambIt) { 
    DTChamberId chId = (*chambIt)->id();
    uint32_t indexCh = chId.rawId();
    int wheel = chId.wheel();
    int sector = chId.sector();
    int station = chId.station();
    
    // ** Compute corrected values and write them to file or database **
    vector<float> newDelays;

    // **  Retrieve Delays Loaded in MiniCrates ** 
    if(readOldFromDb) {    // read from db 
      DTConfigPedestals *pedestals = dtConfig->getDTConfigPedestals();
      const DTLayer *layer = muonGeom->layer(DTLayerId(chId,1,1));
      float delay = pedestals->getOffset(DTWireId(layer->id(),layer->specificTopology().firstChannel())); 
      coarseDelay = int(delay/25.);
      oldFineDelay = delay - coarseDelay * 25.;
    }
    else {                 // read from map created from txt file
      coarseDelay = oldDelayMap[chId].first;
      oldFineDelay = oldDelayMap[chId].second;
    }

    // ** Retrieve t0Mean histograms **
    TH1F *t0H = getHisto<TH1F>(dbe->get(getMEName(t0MeanHistoTag,"", chId)));
    float newFineDelay = -999;   // initialize to dummy number
    cout <<"MG: " << getMEName(t0MeanHistoTag,"", chId) << " entries: " << t0H->GetEntries() << endl; 
    if (t0H->GetEntries() > minEntries) {
      Double_t mean;
      // ** Find Mean Value of the distribution ** 
      if(gaussMean) {
	TF1 *funct = t0H->GetFunction("gaus");
	mean = funct->GetParameter(1);
      }
      else {
	mean = t0H->GetMean();
      }
      
      // ** Retrieve Worst Phase values **
      int wpCoarseDelay;     
      float wpFineDelay;
      worstPhaseMap->get(chId, wpCoarseDelay,  wpFineDelay, DTTimeUnits::ns);
//       cout << "wpFineDelay, oldFineDelay, mean: " << wpFineDelay << " " 
// 	   << oldFineDelay << " " << mean << endl;
      float bpFineDelay = (wpFineDelay < 12.5)? (wpFineDelay + 12.5) : (wpFineDelay - 12.5);  // Best Phase: half BX far from the worst phase 
      // ** Calculate correction **
      float diffFineDelays  = oldFineDelay + (mean - bpFineDelay); // positive mean shift implies positive delay correction
      int bxDiff = (int) (diffFineDelays / 25);
      coarseDelay += bxDiff;
      newFineDelay = fmodf(diffFineDelays, 25);
//       cout << "diffFineDelays, newFineDelay, bxDiff, coarseDelay: " << diffFineDelays 
// 	   << " "<< newFineDelay << " " << bxDiff << " " << coarseDelay << endl;
    }
    else {
      LogProblem(category()) << "[" << testName << "Test]:  Not enough entries in hist for Chamber "  
			     << indexCh << endl;
    }

    newDelays.push_back(wheel);
    newDelays.push_back(sector);
    newDelays.push_back(station);
    newDelays.push_back(coarseDelay);
    newDelays.push_back(newFineDelay);    
    pair< DTChamberId,  vector<float> > chDelays;
    chDelays.first = chId; 
    chDelays.second = newDelays;
    delayMap.insert(chDelays);
   }
}

void DTFineDelayCorr::endJob(){

  DTLocalTriggerBaseTest::endJob();

   if (writeDB) {
     // to be added if needed
   }
   else { // write txt file
     // ** Open output file **
     ofstream outFile(outputFileName.c_str());
     for(map< DTChamberId,  vector<float> >::const_iterator iter = delayMap.begin();
	 iter != delayMap.end() ; iter++) {
       // writing
       ostream_iterator<float> oit(outFile, " ");    
       copy((*iter).second.begin(), (*iter).second.end(), oit);
       outFile << endl;
     }   
     outFile.close();
   }
}
