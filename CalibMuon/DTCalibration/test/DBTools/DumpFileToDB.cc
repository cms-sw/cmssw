
/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include <iostream>

#include "DumpFileToDB.h"
#include "DTCalibrationMap.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DTObjects/interface/DTRecoUncertainties.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

using namespace edm;
using namespace std;

DumpFileToDB::DumpFileToDB(const ParameterSet& pset) {

  dbToDump = pset.getUntrackedParameter<string>("dbToDump", "TTrigDB");

  if(dbToDump != "ChannelsDB")
    theCalibFile = new DTCalibrationMap(pset.getUntrackedParameter<ParameterSet>("calibFileConfig"));

  mapFileName = pset.getUntrackedParameter<ParameterSet>("calibFileConfig").getUntrackedParameter<string>("calibConstFileName", "dummy.txt");

  if(dbToDump != "VDriftDB" &&
     dbToDump != "TTrigDB" &&
     dbToDump != "TZeroDB" && 
     dbToDump != "NoiseDB" &&
     dbToDump != "DeadDB" &&
     dbToDump != "ChannelsDB" &&
     dbToDump != "RecoUncertDB")
    cout << "[DumpFileToDB] *** Error: parameter dbToDump is not valid, check the cfg file" << endl;

  diffMode = pset.getUntrackedParameter<bool>("differentialMode", false);
  if(diffMode) {
    if(dbToDump != "TTrigDB") {
      cout << "***Error: differential mode currentl implemented for ttrig only" << endl;
      abort();
    }
    cout << "Using differential mode: mean value of txt table will be added to the current DB value" << endl;
  }

}
 
DumpFileToDB::~DumpFileToDB(){}


void DumpFileToDB::endJob() {
  if(dbToDump == "VDriftDB") { // Write the Vdrift

    // Create the object to be written to DB
    DTMtime* mtime = new DTMtime();

    // Loop over file entries
    for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	keyAndCalibs != theCalibFile->keyAndConsts_end();
	++keyAndCalibs) {
      cout << "key: " << (*keyAndCalibs).first
	   << " vdrift (cm/ns): " << theCalibFile->meanVDrift((*keyAndCalibs).first)
	   << " hit reso (cm): " << theCalibFile->sigma_meanVDrift((*keyAndCalibs).first) << endl;
      // vdrift is cm/ns , resolution is cm
      mtime->set((*keyAndCalibs).first.superlayerId(),
		 theCalibFile->meanVDrift((*keyAndCalibs).first), 
		 theCalibFile->sigma_meanVDrift((*keyAndCalibs).first),
		 DTVelocityUnits::cm_per_ns);
    }

    cout << "[DumpFileToDB]Writing mtime object to DB!" << endl;
    string record = "DTMtimeRcd";
    DTCalibDBUtils::writeToDB<DTMtime>(record, mtime);

  } else if(dbToDump == "TTrigDB") { // Write the TTrig

    // Create the object to be written to DB
    DTTtrig* tTrig = new DTTtrig();

    // Loop over file entries
    for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	keyAndCalibs != theCalibFile->keyAndConsts_end();
	++keyAndCalibs) {
      

      
      if(diffMode) { // sum the correction in the txt file (for the mean value) to what is input DB 

	// retrieve the previous value from the DB
        float tmean;
        float trms;
        float kFactor;
	// ttrig and rms are ns
        tTrigMapOrig->get((*keyAndCalibs).first.rawId(), tmean, trms, kFactor, DTTimeUnits::ns);
	
	if(theCalibFile->kFactor((*keyAndCalibs).first) != 0 || kFactor != 0) {
	  cout << "***Error: the differentialMode can only be used with kFactor = 0, old: " << kFactor
	       << " new: " << theCalibFile->kFactor((*keyAndCalibs).first) << endl;
	  abort();
	}

	tTrig->set((*keyAndCalibs).first.superlayerId(),
		   theCalibFile->tTrig((*keyAndCalibs).first) + tmean, 
		   theCalibFile->sigma_tTrig((*keyAndCalibs).first),
		   theCalibFile->kFactor((*keyAndCalibs).first),
		   DTTimeUnits::ns);
	
	cout << "key: " << (*keyAndCalibs).first
	     << " ttrig_mean (ns): " << theCalibFile->tTrig((*keyAndCalibs).first) + tmean
	     << " ttrig_sigma(ns): " << theCalibFile->sigma_tTrig((*keyAndCalibs).first)
	     << " kFactor: " << theCalibFile->kFactor((*keyAndCalibs).first) << endl;

      } else {
	cout << "key: " << (*keyAndCalibs).first
	     << " ttrig_mean (ns): " << theCalibFile->tTrig((*keyAndCalibs).first)
	     << " ttrig_sigma(ns): " << theCalibFile->sigma_tTrig((*keyAndCalibs).first)
	     << " kFactor: " << theCalibFile->kFactor((*keyAndCalibs).first) << endl;


	tTrig->set((*keyAndCalibs).first.superlayerId(),
		   theCalibFile->tTrig((*keyAndCalibs).first), 
		   theCalibFile->sigma_tTrig((*keyAndCalibs).first),
		   theCalibFile->kFactor((*keyAndCalibs).first),
		   DTTimeUnits::ns);
      }
    }

    cout << "[DumpFileToDB]Writing ttrig object to DB!" << endl;
    string record = "DTTtrigRcd";
    DTCalibDBUtils::writeToDB<DTTtrig>(record, tTrig);

  } else if(dbToDump == "TZeroDB") { // Write the T0

    // Create the object to be written to DB
    DTT0* tZeroMap = new DTT0();

    // Loop over file entries
    for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	keyAndCalibs != theCalibFile->keyAndConsts_end();
	++keyAndCalibs) {
      float t0mean = (*keyAndCalibs).second[5];
      float t0rms = (*keyAndCalibs).second[6];
      cout << "key: " << (*keyAndCalibs).first
	   << " T0 mean (TDC counts): " << t0mean
	   << " T0_rms (TDC counts): " << t0rms << endl;
      tZeroMap->set((*keyAndCalibs).first,
		    t0mean,
		    t0rms,
		    DTTimeUnits::counts);
    }

    cout << "[DumpFileToDB]Writing tZero object to DB!" << endl;
    string record = "DTT0Rcd";
    DTCalibDBUtils::writeToDB<DTT0>(record, tZeroMap);

  } else if(dbToDump == "NoiseDB") { // Write the Noise
    DTStatusFlag *statusMap = new DTStatusFlag();
    
    // Loop over file entries
    for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	keyAndCalibs != theCalibFile->keyAndConsts_end();
	++keyAndCalibs) {
      cout << "key: " << (*keyAndCalibs).first
	   << " Noisy flag: " << (*keyAndCalibs).second[7] << endl;
      statusMap->setCellNoise((*keyAndCalibs).first,
			      (*keyAndCalibs).second[7]);
    }

    cout << "[DumpFileToDB]Writing Noise Map object to DB!" << endl;
    string record = "DTStatusFlagRcd";
    DTCalibDBUtils::writeToDB<DTStatusFlag>(record, statusMap);
  
  }else if(dbToDump == "DeadDB") { // Write the tp-dead
    DTDeadFlag *deadMap = new DTDeadFlag();
    
    // Loop over file entries
    for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	keyAndCalibs != theCalibFile->keyAndConsts_end();
	++keyAndCalibs) {
      cout << "key: " << (*keyAndCalibs).first
	   << " dead flag: " << (*keyAndCalibs).second[7] << endl;
      deadMap->setCellDead_TP((*keyAndCalibs).first,
			      (*keyAndCalibs).second[7]);
    }

    cout << "[DumpFileToDB]Writing Noise Map object to DB!" << endl;
    string record = "DTDeadFlagRcd";
    DTCalibDBUtils::writeToDB<DTDeadFlag>(record, deadMap);
  
  } else if (dbToDump == "ChannelsDB") { //Write channels map
    
    DTReadOutMapping* ro_map = new DTReadOutMapping( "cmssw_ROB",
                                                     "cmssw_ROS" );
    //Loop over file entries
    string line;
    ifstream file(mapFileName.c_str());
    while (getline(file,line)) {
      if( line == "" || line[0] == '#' ) continue; // Skip comments and empty lines
      stringstream linestr;
      linestr << line;
      vector <int> channelMap = readChannelsMap(linestr);
      int status = ro_map->insertReadOutGeometryLink(channelMap[0],
						     channelMap[1],
						     channelMap[2],
						     channelMap[3],
						     channelMap[4],
						     channelMap[5],
						     channelMap[6],
						     channelMap[7],
						     channelMap[8],
						     channelMap[9],
						     channelMap[10]);
      cout << "ddu " << channelMap[0] << " "
	   << "ros " << channelMap[1] << " "
	   << "rob " << channelMap[2] << " "
	   << "tdc " << channelMap[3] << " "
	   << "channel " << channelMap[4] << " "
	   << "wheel " << channelMap[5] << " "
	   << "station " << channelMap[6] << " "
	   << "sector " << channelMap[7] << " "
	   << "superlayer " << channelMap[8] << " "
	   << "layer " << channelMap[9] << " "
	   << "wire " << channelMap[10] << " " << "  -> ";                
      cout << "insert status: " << status << std::endl;
    }
    string record = "DTReadOutMappingRcd";
    DTCalibDBUtils::writeToDB<DTReadOutMapping>(record, ro_map);
  } else if(dbToDump == "RecoUncertDB") { // Write the Uncertainties

    // Create the object to be written to DB
    DTRecoUncertainties* uncert = new DTRecoUncertainties();

    // FIXME: should come from the configuration; to be changed whenever a new schema for values is introduced.
    uncert->setVersion(1); // Uniform uncertainties per SL and step; parameters 0-3 are for steps 1-4.

    // Loop over file entries
    for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	keyAndCalibs != theCalibFile->keyAndConsts_end();
	++keyAndCalibs) {

      //DTSuperLayerId slId = (*keyAndCalibs).first.superlayerId();
      
      vector<float> values = (*keyAndCalibs).second;

      vector<float> uncerts(values.begin()+8, values.end());

      uncert->set((*keyAndCalibs).first, uncerts);

      cout << endl;
      

    }

    cout << "[DumpFileToDB]Writing RecoUncertainties object to DB!" << endl;
    string record = "DTRecoUncertaintiesRcd";
    DTCalibDBUtils::writeToDB<DTRecoUncertainties>(record, uncert);
  }
}



  
vector <int> DumpFileToDB::readChannelsMap (stringstream &linestr) {
  //The hardware channel
  int ddu_id = 0;
  int ros_id = 0;
  int rob_id = 0;
  int tdc_id = 0;
  int channel_id = 0;
  //The software channel
  int wheel_id = 0;
  int station_id = 0;
  int sector_id = 0;
  int superlayer_id = 0;
  int layer_id = 0;
  int wire_id = 0;

  linestr  >> ddu_id
	   >> ros_id
	   >> rob_id
	   >> tdc_id
	   >> channel_id
	   >> wheel_id
	   >> station_id
	   >> sector_id
	   >> superlayer_id
	   >> layer_id
	   >> wire_id;
    
  vector<int> channelMap;
  channelMap.push_back(ddu_id);
  channelMap.push_back(ros_id);
  channelMap.push_back(rob_id);
  channelMap.push_back(tdc_id);
  channelMap.push_back(channel_id);
  channelMap.push_back(wheel_id);
  channelMap.push_back(station_id);
  channelMap.push_back(sector_id);
  channelMap.push_back(superlayer_id);
  channelMap.push_back(layer_id);
  channelMap.push_back(wire_id);
  return channelMap;
}


void DumpFileToDB::beginRun(const edm::Run& run, const edm::EventSetup& setup ) {

  if(diffMode) {
    if(dbToDump == "TTrigDB") { // read the original DB
      ESHandle<DTTtrig> tTrig;
      setup.get<DTTtrigRcd>().get(tTrig);
      tTrigMapOrig = &*tTrig;
      cout << "[DumpDBToFile] TTrig version: " << tTrig->version() << endl;
    }
  }
}
