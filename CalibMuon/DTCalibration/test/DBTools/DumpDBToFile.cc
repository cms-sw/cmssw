
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/01/23 13:55:12 $
 *  $Revision: 1.5 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DumpDBToFile.h"
#include "DTCalibrationMap.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

using namespace edm;
using namespace std;

DumpDBToFile::DumpDBToFile(const ParameterSet& pset) {
  theCalibFile = new DTCalibrationMap(pset.getUntrackedParameter<ParameterSet>("calibFileConfig"));
  theOutputFileName = pset.getUntrackedParameter<string>("outputFileName");

  dbToDump = pset.getUntrackedParameter<string>("dbToDump", "TTrigDB");

  if(dbToDump != "TTrigDB" && dbToDump != "TZeroDB" && dbToDump != "NoiseDB")
    cout << "[DumpDBToFile] *** Error: parameter dbToDump is not valid, check the cfg file" << endl;

}

DumpDBToFile::~DumpDBToFile(){}


void DumpDBToFile::beginJob(const EventSetup& setup) {
  // Read the right DB accordingly to the parameter dbToDump
  if(dbToDump == "TTrigDB") {
    ESHandle<DTTtrig> tTrig;
    setup.get<DTTtrigRcd>().get(tTrig);
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
  }
}


void DumpDBToFile::endJob() {
  
  static const double convToNs = 25./32.;
  if(dbToDump == "TTrigDB") {
    for(DTTtrig::const_iterator ttrig = tTrigMap->begin();
	ttrig != tTrigMap->end(); ttrig++) {
      cout << "Wh: " << (*ttrig).first.wheelId
	   << " St: " << (*ttrig).first.stationId
	   << " Sc: " << (*ttrig).first.sectorId
	   << " Sl: " << (*ttrig).first.slId
	   << " TTrig mean (ns): " << (*ttrig).second.tTrig * convToNs
	   << " TTrig sigma (ns): " << (*ttrig).second.tTrms * convToNs<< endl;

      DTWireId wireId((*ttrig).first.wheelId,
		      (*ttrig).first.stationId,
		      (*ttrig).first.sectorId,
		      (*ttrig).first.slId, 0, 0);
      vector<float> consts;
      consts.push_back((*ttrig).second.tTrig * convToNs);
      consts.push_back((*ttrig).second.tTrms * convToNs);
      consts.push_back(-1);
      consts.push_back(-1);

      theCalibFile->addCell(wireId, consts);
    }
  } else if(dbToDump == "TZeroDB") {
    for(DTT0::const_iterator tzero = tZeroMap->begin();
	tzero != tZeroMap->end(); tzero++) {
      DTWireId wireId((*tzero).first.wheelId,
		      (*tzero).first.stationId,
		      (*tzero).first.sectorId,
		      (*tzero).first.slId,
		      (*tzero).first.layerId,
		      (*tzero).first.cellId);
      cout << wireId
	   << " TZero mean (TDC counts): " << (*tzero).second.t0mean
	   << " TZero RMS (TDC counts): " << (*tzero).second.t0rms << endl;
      vector<float> consts;
      consts.push_back(-1);
      consts.push_back(-1);
      consts.push_back(-1);
      consts.push_back(-1);
      consts.push_back((*tzero).second.t0mean);      
      consts.push_back((*tzero).second.t0rms);

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
      consts.push_back(-9999999);      
      consts.push_back(-9999999);
      consts.push_back((*statusFlag).second.noiseFlag);

      theCalibFile->addCell(wireId, consts);
    }
  }

  theCalibFile->writeConsts(theOutputFileName);
}

