/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/05/11 17:17:17 $
 *  $Revision: 1.4 $
 *  \author S. Bolognesi
 */

#include "CalibMuon/DTCalibration/plugins/DTTPDeadWriter.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"

/* C++ Headers */
#include <vector> 
#include <set>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "TFile.h"
#include "TH1.h"

using namespace std;
using namespace edm;


// Constructor
DTTPDeadWriter::DTTPDeadWriter(const ParameterSet& pset) {
  // get selected debug option
  debug = pset.getUntrackedParameter<bool>("debug", false); 

  // Create the object to be written to DB
  tpDeadList = new DTDeadFlag();

  if(debug)
    cout << "[DTTPDeadWriter]Constructor called!" << endl;
}

// Destructor
DTTPDeadWriter::~DTTPDeadWriter(){
  if(debug)
    cout << "[DTTPDeadWriter]Destructor called!" << endl;
}

void DTTPDeadWriter::beginRun(const edm::Run&, const EventSetup& setup) {
   // Get the t0 map  
   ESHandle<DTT0> t0;
   setup.get<DTT0Rcd>().get(t0);
   tZeroMap = &*t0;

   // Get the muon Geometry  
   setup.get<MuonGeometryRecord>().get(muonGeom);
}

// Do the job
void DTTPDeadWriter::analyze(const Event & event, const EventSetup& eventSetup) {
  set<DTLayerId> analyzedLayers;

  //Loop on tzero map
  for(DTT0::const_iterator tzero = tZeroMap->begin();
      tzero != tZeroMap->end(); tzero++){

    //Consider what layers have been already considered
// @@@ NEW DTT0 FORMAT
//    DTLayerId layerId = (DTWireId((*tzero).first.wheelId,
//				  (*tzero).first.stationId,
//				  (*tzero).first.sectorId,
//				  (*tzero).first.slId,
//				  (*tzero).first.layerId,
//				  (*tzero).first.cellId)).layerId();
    int channelId = tzero->channelId;
    if ( channelId == 0 ) continue;
    DTLayerId layerId = (DTWireId(channelId)).layerId();
// @@@ NEW DTT0 END
    if(analyzedLayers.find(layerId)==analyzedLayers.end()){
      analyzedLayers.insert(layerId);

      //Take the layer topology
      const DTTopology& dtTopo = muonGeom->layer(layerId)->specificTopology();
      const int firstWire = dtTopo.firstChannel();
      //const int lastWire = dtTopo.lastChannel();
      const int nWires = muonGeom->layer(layerId)->specificTopology().channels();

      //Loop on wires
      for(int wire=firstWire; wire<=nWires; wire++){
	DTWireId wireId(layerId,wire);
	float t0 = 0;
	float t0rms = 0;
	tZeroMap->get(wireId,
		      t0,
		      t0rms,
		      DTTimeUnits::ns);

	//If no t0 stored then is a tp dead channel
	if(!t0){
	  tpDeadList->setCellDead_TP(wireId, true);
	  cout<<"Wire id "<<wireId<<" is TP dead"<<endl;	  
	}
      }
    }
  }
}

// Write objects to DB
void DTTPDeadWriter::endJob() {
  if(debug) 
	cout << "[DTTPDeadWriter]Writing ttrig object to DB!" << endl;

  // FIXME: to be read from cfg?
  string deadRecord = "DTDeadFlagRcd";
  
  // Write the object to DB
  DTCalibDBUtils::writeToDB(deadRecord, tpDeadList);

}  

