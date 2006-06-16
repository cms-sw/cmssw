
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "DumpDBToFile.h"
#include "DTCalibrationFile.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"


using namespace edm;
using namespace std;

DumpDBToFile::DumpDBToFile(const ParameterSet& pset) {
  theCalibFile = new DTCalibrationFile(pset.getUntrackedParameter<ParameterSet>("calibFileConfig"));
  theOutputFileName = pset.getUntrackedParameter<string>("outputFileName");
}

DumpDBToFile::~DumpDBToFile(){}


void DumpDBToFile::beginJob(const EventSetup& setup) {
  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(tTrig);
  tTrigMap = &*tTrig;
  
  cout << "[DumpDBToFile] TTrig version: " << tTrig->version() << endl;

}


void DumpDBToFile::endJob() {
  
  static const double convToNs = 25./32.;
  for(DTTtrig::const_iterator ttrig = tTrigMap->begin();
      ttrig != tTrigMap->end(); ttrig++) {
    cout << "Wh: " << (*ttrig).wheelId
	 << " St: " << (*ttrig).stationId
	 << " Sc: " << (*ttrig).sectorId
	 << " Sl: " << (*ttrig).slId
	 << " TTrig mean (ns): " << (*ttrig).tTrig * convToNs
	 << " TTrig sigma (ns): " << (*ttrig).tTrms * convToNs<< endl;

    DTWireId wireId((*ttrig).wheelId, (*ttrig).stationId, (*ttrig).sectorId, (*ttrig).slId, 0, 0);
    vector<float> consts;
    consts.push_back((*ttrig).tTrig * convToNs);
    consts.push_back((*ttrig).tTrms * convToNs);
    consts.push_back(-1);
    consts.push_back(-1);

    theCalibFile->addCell(wireId, consts);
  }

  theCalibFile->writeConsts(theOutputFileName);
}

