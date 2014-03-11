
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

  if(dbToDump != "VDriftDB" && dbToDump != "TTrigDB" && dbToDump != "TZeroDB" 
     && dbToDump != "NoiseDB" && dbToDump != "DeadDB" && dbToDump != "ChannelsDB" && dbToDump != "RecoUncertDB")
    cout << "[DumpDBToFile] *** Error: parameter dbToDump is not valid, check the cfg file" << endl;
}

DumpDBToFile::~DumpDBToFile(){}


void DumpDBToFile::beginRun(const edm::Run&, const EventSetup& setup) {
  // Read the right DB accordingly to the parameter dbToDump
  if(dbToDump == "VDriftDB") {
    ESHandle<DTMtime> mTime;
    setup.get<DTMtimeRcd>().get(mTime);
    mTimeMap = &*mTime;
    cout << "[DumpDBToFile] MTime version: " << mTime->version() << endl;
  } else if(dbToDump == "TTrigDB") {
    ESHandle<DTTtrig> tTrig;
    setup.get<DTTtrigRcd>().get(dbLabel,tTrig);
    tTrigMap = &*tTrig;
    cout << "[DumpDBToFile] TTrig version: " << tTrig->version() << endl;
  } else if(dbToDump == "TZeroDB") {
    ESHandle<DTT0> t0;
    setup.get<DTT0Rcd>().get(t0);
    tZeroMap = &*t0;
    cout << "[DumpDBToFile] T0 version: " << t0->version() << endl;
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
    ESHandle<DTRecoUncertainties> uncerts;
    setup.get<DTRecoUncertaintiesRcd>().get(uncerts);
    uncertMap = &*uncerts;
    
  }
}


void DumpDBToFile::endJob() {
  
  if (dbToDump != "ChannelsDB") {
    if(dbToDump == "VDriftDB") {
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
	cout << "Wh: " << (*mtime).first.wheelId
	     << " St: " << (*mtime).first.stationId
	     << " Sc: " << (*mtime).first.sectorId
	     << " Sl: " << (*mtime).first.slId
	     << " VDrift (cm/ns): " << vdrift
	     << " Hit reso (cm): " << reso << endl;
	vector<float> consts;
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(vdrift);
	consts.push_back(reso);

	theCalibFile->addCell(wireId, consts);
      }
    } else if(dbToDump == "TTrigDB") {
       for(DTTtrig::const_iterator ttrig = tTrigMap->begin();
	   ttrig != tTrigMap->end(); ttrig++) {
	 DTWireId wireId((*ttrig).first.wheelId,
			 (*ttrig).first.stationId,
			(*ttrig).first.sectorId,
			(*ttrig).first.slId, 0, 0);
        float tmea;
        float trms;
        float kFactor;

        DetId detId(wireId.rawId());
	// ttrig and rms are ns
        tTrigMap->get(detId, tmea, trms, kFactor, DTTimeUnits::ns);
	cout << "Wh: " << (*ttrig).first.wheelId
	     << " St: " << (*ttrig).first.stationId
	     << " Sc: " << (*ttrig).first.sectorId
	     << " Sl: " << (*ttrig).first.slId
	     << " TTrig mean (ns): " << tmea
	     << " TTrig sigma (ns): " << trms << endl;
	vector<float> consts;
	consts.push_back(tmea);
	consts.push_back(trms);
	consts.push_back(kFactor);
	consts.push_back(-1);
	consts.push_back(-1);

	theCalibFile->addCell(wireId, consts);
      }
    } else if(dbToDump == "TZeroDB") {
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
    } else if(dbToDump == "RecoUncertDB") {
      cout << "RecoUncertDB version: " << uncertMap->version() << endl;
      for(DTRecoUncertainties::const_iterator wireAndUncerts = uncertMap->begin();
	  wireAndUncerts != uncertMap->end(); wireAndUncerts++) {
	DTWireId wireId((*wireAndUncerts).first);
	vector<float> values = (*wireAndUncerts).second;
	
	cout << wireId;
	copy(values.begin(), values.end(), ostream_iterator<float>(cout, " cm, "));
	cout << endl;

	vector<float> consts;
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-1);
	consts.push_back(-9999999);      
	consts.push_back(-9999999);
	consts.push_back(-1);
	consts.insert(consts.end(), values.begin(), values.end());

	theCalibFile->addCell(wireId, consts);
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

