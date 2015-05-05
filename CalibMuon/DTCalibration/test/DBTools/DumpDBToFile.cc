
/*
 *  See header file for a description of this class.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "DumpDBToFile.h"
#include "DTCalibrationMap.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DataRecord/interface/DTRecoUncertaintiesRcd.h"
#include "CondFormats/DTObjects/interface/DTRecoUncertainties.h"
#include "CondFormats/DTObjects/interface/DTRecoConditions.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsTtrigRcd.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsVdriftRcd.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsUncertRcd.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>

using namespace edm;
using namespace std;

DumpDBToFile::DumpDBToFile(const ParameterSet& pset) {
  theCalibFile = new DTCalibrationMap(pset.getUntrackedParameter<ParameterSet>("calibFileConfig"));
  theOutputFileName = pset.getUntrackedParameter<string>("outputFileName");

  dbToDump = pset.getUntrackedParameter<string>("dbToDump", "TTrigDB");
  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");
  format = pset.getUntrackedParameter<string>("dbFormat", "Legacy");

  if(dbToDump != "VDriftDB" && dbToDump != "TTrigDB" && dbToDump != "TZeroDB" 
     && dbToDump != "NoiseDB" && dbToDump != "DeadDB" && dbToDump != "ChannelsDB" && dbToDump != "RecoUncertDB")
    throw cms::Exception("IncorrectSetup") << "Parameter dbToDump is not valid, check the cfg file" << endl;

  if (format != "Legacy" &&
      format != "DTRecoConditions")
    throw cms::Exception("IncorrectSetup") << "Parameter format is not valid, check the cfg file" << endl;

}

DumpDBToFile::~DumpDBToFile(){}


void DumpDBToFile::beginRun(const edm::Run&, const EventSetup& setup) {
  // Read the right DB accordingly to the parameter dbToDump
  if(dbToDump == "VDriftDB") {
    if (format=="Legacy") {
      ESHandle<DTMtime> mTime;
      setup.get<DTMtimeRcd>().get(mTime);
      mTimeMap = &*mTime;
    } else if (format=="DTRecoConditions"){
      ESHandle<DTRecoConditions> h_rconds;
      setup.get<DTRecoConditionsVdriftRcd>().get(h_rconds);
      rconds = &*h_rconds;
    }
  } else if(dbToDump == "TTrigDB") {
    if (format=="Legacy") {
      ESHandle<DTTtrig> tTrig;
      setup.get<DTTtrigRcd>().get(dbLabel,tTrig);
      tTrigMap = &*tTrig;
    } else if (format=="DTRecoConditions"){
      ESHandle<DTRecoConditions> h_rconds;
      setup.get<DTRecoConditionsTtrigRcd>().get(h_rconds);
      rconds = &*h_rconds;
    }
  } else if(dbToDump == "TZeroDB") {
    ESHandle<DTT0> t0;
    setup.get<DTT0Rcd>().get(t0);
    tZeroMap = &*t0;
  } else if(dbToDump == "NoiseDB") {
    ESHandle<DTStatusFlag> status;
    setup.get<DTStatusFlagRcd>().get(status);
    statusMap = &*status;
  } else if(dbToDump == "DeadDB") {
    ESHandle<DTDeadFlag> dead;
    setup.get<DTDeadFlagRcd>().get(dead);
    deadMap = &*dead;
  } else if (dbToDump == "ChannelsDB") {
    ESHandle<DTReadOutMapping> channels;
    setup.get<DTReadOutMappingRcd>().get(channels);
    channelsMap = &*channels;
  } else if (dbToDump == "RecoUncertDB") {
    if (format=="Legacy") {
      ESHandle<DTRecoUncertainties> uncerts;
      setup.get<DTRecoUncertaintiesRcd>().get(uncerts);
      uncertMap = &*uncerts;
    } else if (format=="DTRecoConditions"){
      ESHandle<DTRecoConditions> h_rconds;
      setup.get<DTRecoConditionsUncertRcd>().get(h_rconds);
      rconds = &*h_rconds;
    }
  }
}


void DumpDBToFile::endJob() {
  
  if (dbToDump != "ChannelsDB") {

    //---------- VDrifts
    if(dbToDump == "VDriftDB") {
      if (format=="Legacy") {
	int version = 1;
	int type =1; //ie constant
	int nfields=1;
	cout << "[DumpDBToFile] MTime version: " << mTimeMap->version() << endl;
	for(DTMtime::const_iterator mtime = mTimeMap->begin();
	    mtime != mTimeMap->end(); mtime++) {
	  DTWireId wireId((*mtime).first.wheelId,
			  (*mtime).first.stationId,
			  (*mtime).first.sectorId,
			  (*mtime).first.slId, 0, 0);
	  float vdrift;
	  float reso;
	  DetId detId( wireId.rawId() );
	  // vdrift is cm/ns , resolution is cm
	  mTimeMap->get(detId, vdrift, reso, DTVelocityUnits::cm_per_ns);
// 	  cout << "Wh: " << (*mtime).first.wheelId
// 	       << " St: " << (*mtime).first.stationId
// 	       << " Sc: " << (*mtime).first.sectorId
// 	       << " Sl: " << (*mtime).first.slId
// 	       << " VDrift (cm/ns): " << vdrift
// 	       << " Hit reso (cm): " << reso << endl;
	  vector<float> consts = {-1, -1, -1, vdrift, reso, -1, -1, -1, -1, -1, float(version*1000+type*100+nfields), vdrift};
	  theCalibFile->addCell(wireId, consts);
	}
      }	else if (format=="DTRecoConditions") {
	int version = rconds->version() ;
	int type =1; // i.e. expr = "[0]"
	string expr = rconds->getFormulaExpr();
	cout << "[DumpDBToFile] DTRecoConditions (vdrift) version: " << version
	     << " expression: " << expr << endl;
	if (version!=1 || expr!="[0]") throw cms::Exception("Configuration") << "only version 1, type 1 is presently supported for VDriftDB";
	for(DTRecoConditions::const_iterator irc = rconds->begin(); irc != rconds->end(); ++irc) {
	  DTWireId wireId(irc->first);
	  const vector<double>& data = irc->second;
	  int nfields = data.size(); // FIXME check size
	  float vdrift = data[0];
	  float reso = 0;
	  vector<float> consts(11+nfields,-1);
	  consts[3] = vdrift;
	  consts[4] = reso;
	  consts[10] = float(version*1000+type*100+nfields);
	  std::copy(data.begin(),data.end(),consts.begin()+11);
	  theCalibFile->addCell(wireId, consts);
	}
      }

    //---------- TTrigs
    } else if(dbToDump == "TTrigDB") {
      if (format=="Legacy") {
	int version = 1;
	int type = 1; //ie constant
	int nfields =1;
	cout << "[DumpDBToFile] TTrig version: " << tTrigMap->version() << endl;
	for(DTTtrig::const_iterator ttrig = tTrigMap->begin();
	    ttrig != tTrigMap->end(); ttrig++) {
	  DTWireId wireId((*ttrig).first.wheelId,
			  (*ttrig).first.stationId,
			  (*ttrig).first.sectorId,
			  (*ttrig).first.slId, 0, 0);
	  DetId detId(wireId.rawId());
	  float tmea;
	  float trms;
	  float kFactor;
	  // ttrig and rms are ns
	  tTrigMap->get(detId, tmea, trms, kFactor, DTTimeUnits::ns);
// 	  cout << "Wh: " << (*ttrig).first.wheelId
// 	       << " St: " << (*ttrig).first.stationId
// 	       << " Sc: " << (*ttrig).first.sectorId
// 	       << " Sl: " << (*ttrig).first.slId
// 	       << " TTrig mean (ns): " << tmea
// 	       << " TTrig sigma (ns): " << trms << endl;

	  // note that in the free fields we write one single time = tmea+trms*kFactor, as per current use:
	  // https://github.com/cms-sw/cmssw/blob/CMSSW_7_5_X/CalibMuon/DTDigiSync/src/DTTTrigSyncFromDB.cc#L197
	  vector<float> consts = {tmea, trms, kFactor, -1, -1, -1, -1, -1, -1, -1, float(version*1000+type*100+nfields), tmea+ trms*kFactor}; 
	  theCalibFile->addCell(wireId, consts);
	}
      }	else if (format=="DTRecoConditions") {
	int version = rconds->version();
	int type = 1; // i.e. expr = "[0]"
	string expr = rconds->getFormulaExpr();
	if (version!=1||expr!="[0]") throw cms::Exception("Configuration") << "only version 1, type 1 is presently supported for TTrigDB";

	cout << "[DumpDBToFile] DTRecoConditions (ttrig) version: " << rconds->version() 
	     << " expression: " << expr << endl;
	for(DTRecoConditions::const_iterator irc = rconds->begin(); irc != rconds->end(); ++irc) {
	  DTWireId wireId(irc->first);
	  const vector<double>& data = irc->second;
	  int nfields = data.size(); // FIXME check size (should be 1)
	  float ttrig = data[0];
	  float sigma = 0; // Unused in DTRecoConditions
	  float kappa = 0;

	  vector<float> consts(11+nfields,-1);
	  consts[0]=ttrig;
	  consts[1]=sigma;
	  consts[2]=kappa;
	  consts[10] = float(version*1000+type*100+nfields);
	  std::copy(data.begin(),data.end(),consts.begin()+11);
	  theCalibFile->addCell(wireId, consts);	  
	}
      }


    //---------- T0, noise, dead
    } else if(dbToDump == "TZeroDB") {
      cout << "[DumpDBToFile] T0 version: " << tZeroMap->version() << endl;
      for(DTT0::const_iterator tzero = tZeroMap->begin();
	  tzero != tZeroMap->end(); tzero++) {
// @@@ NEW DTT0 FORMAT
//	DTWireId wireId((*tzero).first.wheelId,
//			(*tzero).first.stationId,
//			(*tzero).first.sectorId,
//			(*tzero).first.slId,
//			(*tzero).first.layerId,
//			(*tzero).first.cellId);
        int channelId = tzero->channelId;
        if ( channelId == 0 ) continue;
        DTWireId wireId(channelId);
// @@@ NEW DTT0 END
        float t0mean;
        float t0rms;
        // t0s and rms are TDC counts
        tZeroMap->get(wireId, t0mean, t0rms, DTTimeUnits::counts);
	cout << wireId
	     << " TZero mean (TDC counts): " << t0mean
	     << " TZero RMS (TDC counts): " << t0rms << endl;
	vector<float> consts;
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(t0mean);      
	consts.push_back(t0rms);

	theCalibFile->addCell(wireId, consts);
      }
    } else if(dbToDump == "NoiseDB") {
      for(DTStatusFlag::const_iterator statusFlag = statusMap->begin();
	  statusFlag != statusMap->end(); statusFlag++) {
	DTWireId wireId((*statusFlag).first.wheelId,
			(*statusFlag).first.stationId,
			(*statusFlag).first.sectorId,
			(*statusFlag).first.slId,
			(*statusFlag).first.layerId,
			(*statusFlag).first.cellId);
	cout << wireId
	     << " Noisy Flag: " << (*statusFlag).second.noiseFlag << endl;
	vector<float> consts;
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-9999999);      
	consts.push_back(-9999999);
	consts.push_back((*statusFlag).second.noiseFlag);

	theCalibFile->addCell(wireId, consts);
      }
    }  else if(dbToDump == "DeadDB") {
      for(DTDeadFlag::const_iterator deadFlag = deadMap->begin();
	  deadFlag != deadMap->end(); deadFlag++) {
	DTWireId wireId((*deadFlag).first.wheelId,
			(*deadFlag).first.stationId,
			(*deadFlag).first.sectorId,
			(*deadFlag).first.slId,
			(*deadFlag).first.layerId,
			(*deadFlag).first.cellId);
	cout << wireId
	     << " Dead Flag: " << (*deadFlag).second.dead_TP << endl;
	vector<float> consts;
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-9999999);      
	consts.push_back(-9999999);
	consts.push_back((*deadFlag).second.dead_TP);

	theCalibFile->addCell(wireId, consts);
      }

    //---------- Uncertainties
    } else if(dbToDump == "RecoUncertDB") {
      if (format=="Legacy") {
	int version = 1;
	int type =2; // par[step]
	cout << "RecoUncertDB version: " << uncertMap->version() << endl;
	for(DTRecoUncertainties::const_iterator wireAndUncerts = uncertMap->begin();
	    wireAndUncerts != uncertMap->end(); wireAndUncerts++) {
	  DTWireId wireId((*wireAndUncerts).first);
	  vector<float> values = (*wireAndUncerts).second;	
// 	  cout << wireId;
// 	  copy(values.begin(), values.end(), ostream_iterator<float>(cout, " cm, "));
// 	  cout << endl;
	  int nfields=values.size();
	  vector<float> consts = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, float(version*1000+type*100+nfields)};	  
	  consts.insert(consts.end(), values.begin(), values.end());
	  theCalibFile->addCell(wireId, consts);
	}
      }	else if (format=="DTRecoConditions") {
	int version = rconds->version();
	string expr = rconds->getFormulaExpr();
	int type = 2; // par[step]
	if (version!=1||expr!="par[step]") throw cms::Exception("Configuration") << "only version 1, type 2 is presently supported for RecoUncertDB";

	cout << "[DumpDBToFile] DTRecoConditions (uncerts) version: " << rconds->version() 
	     << " expression: " << expr << endl;

	for(DTRecoConditions::const_iterator irc = rconds->begin(); irc != rconds->end(); ++irc) {
	  DTWireId wireId(irc->first);
	  const vector<double>& data = irc->second;
	  int nfields = data.size();
	  vector<float> consts(11+nfields,-1);
	  consts[10] = float(version*1000+type*100+nfields);
	  std::copy(data.begin(),data.end(),consts.begin()+11);	  
	  theCalibFile->addCell(wireId, consts);
	}
      }
    }
    //Write constants into file
    theCalibFile->writeConsts(theOutputFileName);
  }

  else if (dbToDump == "ChannelsDB"){
    ofstream out(theOutputFileName.c_str());
    for(DTReadOutMapping::const_iterator roLink = channelsMap->begin();
	roLink != channelsMap->end(); roLink++) {
      out << roLink->dduId << ' ' 
	  << roLink->rosId << ' ' 
	  << roLink->robId << ' ' 
	  << roLink->tdcId << ' ' 
	  << roLink->channelId << ' ' 
	  << roLink->wheelId << ' ' 
	  << roLink->stationId << ' ' 
	  << roLink->sectorId << ' ' 
	  << roLink->slId << ' ' 
	  << roLink->layerId << ' ' 
	  << roLink->cellId <<endl;
       
      cout << "ddu " <<roLink->dduId << ' ' 
	   << "ros " <<roLink->rosId << ' ' 
	   << "rob " <<roLink->robId << ' ' 
	   << "tdc " <<roLink->tdcId << ' ' 
	   << "channel " <<roLink->channelId << ' ' << "->" << ' '
	   << "wheel "   <<roLink->wheelId << ' ' 
	   << "station " <<roLink->stationId << ' ' 
	   << "sector "  <<roLink->sectorId << ' ' 
	   << "superlayer " << roLink->slId << ' ' 
	   << "layer "   << roLink->layerId << ' ' 
	   << "wire "    << roLink->cellId << ' '<<endl;
    }
  }


}

