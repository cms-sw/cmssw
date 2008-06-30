
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTNoiseTask.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include <sstream>
#include <string>

using namespace edm;
using namespace std;



DTNoiseTask::DTNoiseTask(const ParameterSet& ps) : nevents(0) {
  dbe = edm::Service<DQMStore>().operator->();
}




DTNoiseTask::~DTNoiseTask(){}



/// BeginJob
void DTNoiseTask::beginJob(const edm::EventSetup& setup) {
  // Get the geometry
  setup.get<MuonGeometryRecord>().get(muonGeom);
}



/// To reset the MEs
void DTNoiseTask::beginLuminosityBlock(const edm::LuminosityBlock&  lumiSeg,
				       const edm::EventSetup& context) {}


  
/// Analyze
void DTNoiseTask::analyze(const edm::Event& e, const edm::EventSetup& c) {}
  


/// Endjob
void DTNoiseTask::endJob() {}


void DTNoiseTask::bookHistos(DTChamberId chId) {
  // set the folder
  stringstream wheel; wheel << chId.wheel();	
  stringstream station; station << chId.station();	
  stringstream sector; sector << chId.sector();	
//   stringstream superLayer; superLayer << dtSL.superlayer();
  dbe->setCurrentFolder("DT/05-Noise/Wheel" + wheel.str() +
			"/Station" + station.str() +
			"/Sector" + sector.str());

  // Build the histo name
  string histoName = string("NoiseRate")
    + "_W" + wheel.str() 
    + "_St" + station.str() 
    + "_Sec" + sector.str() ;
  //     + "_SL" + superLayer.str(); 
  

//   if (debug) {
    cout<<"[DTNoiseTask]: booking chamber histo:"<<endl;
    cout<<"              folder "<< "DT/01-Noise/Wheel" + wheel.str() +
      "/Station" + station.str() +
      "/Sector" + sector.str() + "/" << endl; 
    cout<<"              histoName "<<histoName<<endl;
//   }

//   if ( readTTrigDB ) 
//     tTrigMap->slTtrig( dtSL, tTrig, tTrigRMS); 
//   else tTrig = defaultTTrig;



  // Get the chamber from the geometry
  int nWires_max = 0;
  const DTChamber* dtchamber = muonGeom->chamber(chId);
  const vector<const DTSuperLayer*> superlayers = dtchamber->superLayers();

  // Loop over layers and find the max # of wires
  for(vector<const DTSuperLayer*>::const_iterator sl = superlayers.begin();
      sl != superlayers.end(); ++sl) { // loop over SLs
    vector<const DTLayer*> layers = (*sl)->layers();
    for(vector<const DTLayer*>::const_iterator lay = layers.begin();
	lay != layers.end(); ++lay) { // loop over layers
      int nWires = (*lay)->specificTopology().channels();
      if(nWires > nWires_max) nWires_max = nWires;
    }
  }
  dbe->book2D(histoName,"Noise rate (Hz) per channel", nWires_max,1, nWires_max+1,12,1,13);
}
