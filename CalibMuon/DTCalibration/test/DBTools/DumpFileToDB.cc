
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
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DTObjects/interface/DTRecoUncertainties.h"
#include "CondFormats/DTObjects/interface/DTRecoConditions.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsTtrigRcd.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsVdriftRcd.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsUncertRcd.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

using namespace edm;
using namespace std;

DumpFileToDB::DumpFileToDB(const ParameterSet& pset) {

  dbToDump = pset.getUntrackedParameter<string>("dbToDump", "TTrigDB");
  format = pset.getUntrackedParameter<string>("dbFormat", "Legacy");

  cout << "Writing DB: " << dbToDump << " with format: " << format << endl;


  if(dbToDump != "ChannelsDB") theCalibFile = new DTCalibrationMap(pset.getUntrackedParameter<ParameterSet>("calibFileConfig"));

  mapFileName = pset.getUntrackedParameter<ParameterSet>("calibFileConfig").getUntrackedParameter<string>("calibConstFileName", "dummy.txt");

  if(dbToDump != "VDriftDB" &&
     dbToDump != "TTrigDB" &&
     dbToDump != "TZeroDB" && 
     dbToDump != "NoiseDB" &&
     dbToDump != "DeadDB" &&
     dbToDump != "ChannelsDB" &&
     dbToDump != "RecoUncertDB")
    throw cms::Exception("IncorrectSetup") << "Parameter dbToDump is not valid";

  if (format != "Legacy" &&
      format != "DTRecoConditions")
    throw cms::Exception("IncorrectSetup") << "Parameter format is not valid";

  if (format == "DTRecoConditions" &&
      (dbToDump != "VDriftDB" && 
       dbToDump != "TTrigDB" &&
       dbToDump != "RecoUncertDB"))
    throw cms::Exception("IncorrectSetup") << "DTRecoConditions currently implemented only for TTrigDB, VDriftDB, RecoUncertDB";

  diffMode = pset.getUntrackedParameter<bool>("differentialMode", false);
  if(diffMode) {
    if(dbToDump != "TTrigDB") {
      throw cms::Exception("IncorrectSetup") << "Differential mode currently implemented for ttrig only";
    }
    cout << "Using differential mode: mean value of txt table will be added to the current DB value" << endl;
  }

}
 
DumpFileToDB::~DumpFileToDB(){}


void DumpFileToDB::endJob() {

  //---------- VDrift
  if(dbToDump == "VDriftDB") {
    if (format=="Legacy") {
      DTMtime* mtime = new DTMtime();
      for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	  keyAndCalibs != theCalibFile->keyAndConsts_end();
	  ++keyAndCalibs) {
	// 	cout << "key: " << (*keyAndCalibs).first
	// 	     << " vdrift (cm/ns): " << theCalibFile->meanVDrift((*keyAndCalibs).first)
	// 	     << " hit reso (cm): " << theCalibFile->sigma_meanVDrift((*keyAndCalibs).first) << endl;
	// vdrift is cm/ns , resolution is cm
	mtime->set((*keyAndCalibs).first.superlayerId(),
		   theCalibFile->meanVDrift((*keyAndCalibs).first), 
		   theCalibFile->sigma_meanVDrift((*keyAndCalibs).first),
		   DTVelocityUnits::cm_per_ns);
      }
      DTCalibDBUtils::writeToDB<DTMtime>("DTMtimeRcd", mtime);
    } else if (format=="DTRecoConditions") {
      DTRecoConditions* conds = new DTRecoConditions();
      conds->setFormulaExpr("[0]");
      //      conds->setFormulaExpr("[0]*(1-[1]*x)");
      int version = 1;
      conds->setVersion(version);
      for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	  keyAndCalibs != theCalibFile->keyAndConsts_end();
	  ++keyAndCalibs) {

	vector<float> values = (*keyAndCalibs).second;
	int fversion = int(values[10]/1000);
	int type = (int(values[10])%1000)/100;
	int nfields = int(values[10])%100;
	if (type !=1) throw cms::Exception("IncorrectSetup") << "Input file is not for VDriftDB";
	if (values.size()!=unsigned(nfields+11)) throw cms::Exception("IncorrectSetup") << "Inconsistent number of fields";
	if (fversion!=version) throw cms::Exception("IncorrectSetup") << "Inconsistent version of file";
	
	vector<double> params(values.begin()+11, values.begin()+11+nfields);
	conds->set((*keyAndCalibs).first, params);
      }
      DTCalibDBUtils::writeToDB<DTRecoConditions>("DTRecoConditionsVdriftRcd", conds);      
    }
  
  //---------- TTrig
  } else if(dbToDump == "TTrigDB") {
    if (format=="Legacy") {
      DTTtrig* tTrig = new DTTtrig();
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
	    throw cms::Exception("IncorrectSetup") << "The differentialMode can only be used with kFactor = 0, old: " << kFactor
						   << " new: " << theCalibFile->kFactor((*keyAndCalibs).first);
	  }
	  tTrig->set((*keyAndCalibs).first.superlayerId(),
		     theCalibFile->tTrig((*keyAndCalibs).first) + tmean, 
		     theCalibFile->sigma_tTrig((*keyAndCalibs).first),
		     theCalibFile->kFactor((*keyAndCalibs).first),
		     DTTimeUnits::ns);
// 	  cout << "key: " << (*keyAndCalibs).first
// 	       << " ttrig_mean (ns): " << theCalibFile->tTrig((*keyAndCalibs).first) + tmean
// 	       << " ttrig_sigma(ns): " << theCalibFile->sigma_tTrig((*keyAndCalibs).first)
// 	       << " kFactor: " << theCalibFile->kFactor((*keyAndCalibs).first) << endl;

	} else { // Normal mode
// 	  cout << "key: " << (*keyAndCalibs).first
// 	       << " ttrig_mean (ns): " << theCalibFile->tTrig((*keyAndCalibs).first)
// 	       << " ttrig_sigma(ns): " << theCalibFile->sigma_tTrig((*keyAndCalibs).first)
// 	       << " kFactor: " << theCalibFile->kFactor((*keyAndCalibs).first) << endl;
	  tTrig->set((*keyAndCalibs).first.superlayerId(),
		     theCalibFile->tTrig((*keyAndCalibs).first), 
		     theCalibFile->sigma_tTrig((*keyAndCalibs).first),
		     theCalibFile->kFactor((*keyAndCalibs).first),
		     DTTimeUnits::ns);
	}
      }
      DTCalibDBUtils::writeToDB<DTTtrig>("DTTtrigRcd", tTrig);

    } else if (format=="DTRecoConditions") {
      DTRecoConditions* conds = new DTRecoConditions();
      conds->setFormulaExpr("[0]");
      int version = 1;
      conds->setVersion(version);

      for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	  keyAndCalibs != theCalibFile->keyAndConsts_end();
	  ++keyAndCalibs) {

	vector<float> values = (*keyAndCalibs).second;
	int fversion = int(values[10]/1000);
	int type = (int(values[10])%1000)/100;
	int nfields = int(values[10])%100;

	if (type!=1) throw cms::Exception("IncorrectSetup") << "Only type==1 currently supported for TTrigDB";
	if (values.size()!=unsigned(nfields+11)) throw cms::Exception("IncorrectSetup") << "Inconsistent number of fields";
	if (fversion!=version) throw cms::Exception("IncorrectSetup") << "Inconsistent version of file";

	vector<double> params(values.begin()+11, values.begin()+11+nfields);
	conds->set((*keyAndCalibs).first, params);
      }
      DTCalibDBUtils::writeToDB<DTRecoConditions>("DTRecoConditionsTtrigRcd", conds);
    }
    
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

    DTCalibDBUtils::writeToDB<DTT0>("DTT0Rcd", tZeroMap);

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

    DTCalibDBUtils::writeToDB<DTStatusFlag>("DTStatusFlagRcd", statusMap);
  
  } else if(dbToDump == "DeadDB") { // Write the tp-dead
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

    DTCalibDBUtils::writeToDB<DTDeadFlag>("DTDeadFlagRcd", deadMap);
  
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
    DTCalibDBUtils::writeToDB<DTReadOutMapping>("DTReadOutMappingRcd", ro_map);


  //---------- Uncertainties
  } else if(dbToDump == "RecoUncertDB") { // Write the Uncertainties

    if (format=="Legacy") {
      DTRecoUncertainties* uncert = new DTRecoUncertainties();
      int version = 1; // Uniform uncertainties per SL and step; parameters 0-3 are for steps 1-4.
      uncert->setVersion(version);
      // Loop over file entries
      for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	  keyAndCalibs != theCalibFile->keyAndConsts_end();
	  ++keyAndCalibs) {
	vector<float> values = (*keyAndCalibs).second;
	vector<float> uncerts(values.begin()+11, values.end());
	uncert->set((*keyAndCalibs).first, uncerts);
      }
      DTCalibDBUtils::writeToDB<DTRecoUncertainties>("DTRecoUncertaintiesRcd", uncert);

    } else if (format=="DTRecoConditions") {
      DTRecoConditions* conds = new DTRecoConditions();
      conds->setFormulaExpr("par[step]");
      int version = 1; // Uniform uncertainties per SL and step; parameters 0-3 are for steps 1-4.
      conds->setVersion(version);

      for(DTCalibrationMap::const_iterator keyAndCalibs = theCalibFile->keyAndConsts_begin();
	  keyAndCalibs != theCalibFile->keyAndConsts_end();
	  ++keyAndCalibs) {

	vector<float> values = (*keyAndCalibs).second;
	int fversion = int(values[10]/1000);
	int type = (int(values[10])%1000)/100;
	int nfields = int(values[10])%100;
	if (type !=2) throw cms::Exception("IncorrectSetup") << "Only type==2 supported for uncertainties DB";
	if(values.size()!=unsigned(nfields+11)) throw cms::Exception("IncorrectSetup") << "Inconsistent number of fields";
	if (fversion!=version) throw cms::Exception("IncorrectSetup") << "Inconsistent version of file";

	vector<double> params(values.begin()+11, values.begin()+11+nfields);
	conds->set((*keyAndCalibs).first, params);
      }
      DTCalibDBUtils::writeToDB<DTRecoConditions>("DTRecoConditionsUncertRcd", conds);
    }
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
