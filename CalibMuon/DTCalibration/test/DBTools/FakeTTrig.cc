 /*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 */


#include "CalibMuon/DTCalibration/test/DBTools/FakeTTrig.h"

// Framework
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CalibMuon/DTCalibration/test/DBTools/DTCalibrationMap.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

// Database
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

//Random generator
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"

// DTDigitizer
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"


using namespace std;
using namespace edm;



FakeTTrig::FakeTTrig(const ParameterSet& pset) :
  dataBaseWriteWasDone(false) {

  cout << "[FakeTTrig] Constructor called! " << endl;

  // further configurable smearing
  smearing=pset.getUntrackedParameter<double>("smearing");
  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");

  // get random engine
  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "RandomNumberGeneratorService for DTFakeTTrigDB missing in cfg file";
  }
  ps = pset;
}


FakeTTrig::~FakeTTrig(){
  cout << "[FakeTTrig] Destructor called! " << endl;
}

void FakeTTrig::beginRun(const edm::Run&, const EventSetup& setup) {
  cout << "[FakeTTrig] entered into beginRun! " << endl;
  setup.get<MuonGeometryRecord>().get(muonGeom);

  // Get the tTrig reference map
  if (ps.getUntrackedParameter<bool>("readDB", true)) 
    setup.get<DTTtrigRcd>().get(dbLabel,tTrigMapRef);  
}

void FakeTTrig::beginLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const&) {
  if(!dataBaseWriteWasDone) {
    dataBaseWriteWasDone = true;

    cout << "[FakeTTrig] entered into beginLuminosityBlock! " << endl;
 
    edm::Service<edm::RandomNumberGenerator> rng;
    CLHEP::HepRandomEngine* engine = &rng->getEngine(lumi.index());

    // Get the superlayers and layers list
    vector<DTSuperLayer*> dtSupLylist = muonGeom->superLayers();
    // Create the object to be written to DB
    DTTtrig* tTrigMap = new DTTtrig();

    for (vector<DTSuperLayer*>::const_iterator sl = dtSupLylist.begin();
         sl != dtSupLylist.end(); sl++) {

      // get the time of fly
      double timeOfFly = tofComputation(*sl);
      // get the time of wire propagation
      double timeOfWirePropagation = wirePropComputation(*sl);
      // get the gaussian smearing
      double gaussianSmearing = CLHEP::RandGaussQ::shoot(engine, 0., smearing);
      // get the fake tTrig pedestal
      double pedestral = ps.getUntrackedParameter<double>("fakeTTrigPedestal", 500);

      if ( ps.getUntrackedParameter<bool>("readDB", true) ){
        tTrigMapRef->get((*sl)->id(), tTrigRef, tTrigRMSRef, kFactorRef, DTTimeUnits::ns );
        // pedestral = tTrigRef;
        pedestral = tTrigRef +  kFactorRef*tTrigRMSRef ;
      }

      DTSuperLayerId slId = (*sl)->id();
      // if the FakeTtrig has to be smeared with a Gaussian
      double fakeTTrig = pedestral + timeOfFly + timeOfWirePropagation + gaussianSmearing;
      // if the FakeTtrig is scaled of a number of bunch crossing
      //  double fakeTTrig = pedestral - 75.;
      tTrigMap->set(slId, fakeTTrig, 0,0, DTTimeUnits::ns);
    }

    // Write the object in the DB
    cout << "[FakeTTrig] Writing ttrig object to DB!" << endl;
    string record = "DTTtrigRcd";
    DTCalibDBUtils::writeToDB<DTTtrig>(record, tTrigMap);
  }
}

void FakeTTrig::endJob() { 
  cout << "[FakeTTrig] entered into endJob! " << endl;
}

double FakeTTrig::tofComputation(const DTSuperLayer* superlayer) {

  double tof=0;
  const double cSpeed = 29.9792458; // cm/ns

  if(ps.getUntrackedParameter<bool>("useTofCorrection", true)){
    LocalPoint localPos(0,0,0);
    double flight = superlayer->surface().toGlobal(localPos).mag();
    tof = flight/cSpeed;
  }

  return tof;

}



double FakeTTrig::wirePropComputation(const DTSuperLayer* superlayer) {

  double delay = 0;
  double theVPropWire = ps.getUntrackedParameter<double>("vPropWire", 24.4); // cm/ns

  if(ps.getUntrackedParameter<bool>("useWirePropCorrection", true)){
    DTLayerId lId = DTLayerId(superlayer->id(), 1);
    float halfL  =  superlayer->layer(lId)->specificTopology().cellLenght()/2;
    delay = halfL/theVPropWire;
  }

  return delay;

}

