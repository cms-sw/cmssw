#ifndef L1Trigger_DTUtilities_DTUtilTest_h
#define L1Trigger_DTUtilities_DTUtilTest_h

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"

#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>


#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"
#include "L1Trigger/DTUtilities/interface/DTConfig.h"

#include <iostream>

using namespace std;
using namespace edm;

class DTUtilTest: public EDAnalyzer{
public:
  DTUtilTest(const ParameterSet& pset){
    cout << "constructor executed!!!" << endl;
  }

  ~DTUtilTest(){
    cout << "destructor executed!!!" << endl;
  }

  void analyze(const Event & event, const EventSetup& eventSetup){
    cout << "--- Run: " << event.id().run()
	 << " Event: " << event.id().event() << endl;
    ESHandle<DTGeometry> pDD;
    eventSetup.get<MuonGeometryRecord>().get( pDD );
    
    // check chamber
    for(vector<DTChamber*>::const_iterator det = pDD->chambers().begin();
	det != pDD->chambers().end(); ++det){
      //cout << "Chamber " << (*det)->geographicalId().det() << endl;
      //			  const BoundPlane& surf=(*det)->surface();
      //cout << "surf " << &surf <<  endl;
      cout << "Chamber " << (*det)->id()
	//				  << " Position " << surf.position()
	//				  << " normVect " << surf.normalVector()
	//				  << " bounds W/H/L: " << surf.bounds().width() << "/"
	//				  << surf.bounds().thickness() << "/" << surf.bounds().length()
	   << endl;
      DTConfig config;
      DTTrigGeom MyTG((*det),&config);
      cout <<"TrigGeom wh:" <<  MyTG.wheel()
	   <<" st:" << MyTG.station()
	   <<" se:" << MyTG.sector()
	   << endl;

    }
  }
  
private:
};
 
#endif

