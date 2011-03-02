/*
 *  \author Anna Cimmino
 */
#include <DQM/RPCMonitorClient/interface/SaveDQMFile.h>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// //DataFormats
// #include <DataFormats/MuonDetId/interface/RPCDetId.h>
// #include "DataFormats/RPCDigi/interface/RPCDigi.h"
// #include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

// // Geometry
// #include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
// #include "Geometry/Records/interface/MuonGeometryRecord.h"

// // DQM
// #include "DQMServices/Core/interface/MonitorElement.h"

// #include <map>
// #include <sstream>
//#include <math.h>

using namespace edm;
using namespace std;

SaveDQMFile::SaveDQMFile(const ParameterSet& ps ){
 
  LogVerbatim ("readFile") << "[SaveDQMFile]: Constructor";

  myFile_= ps.getUntrackedParameter<string>("OutputFile", "uffa.root");
}

SaveDQMFile::~SaveDQMFile(){
  dbe_ = 0;
}

void SaveDQMFile::beginJob(){}

void SaveDQMFile::beginRun(const Run& r, const EventSetup& iSetup){
  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
 }

void SaveDQMFile::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {}

void SaveDQMFile::analyze(const edm::Event& iEvent, const edm::EventSetup& c){}

void SaveDQMFile::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {}
 
void SaveDQMFile::endRun(const Run& r, const EventSetup& c){

  if(dbe_ && myFile_ != "") {
    LogVerbatim ("savedqmfile") << "[SaveDQMFile]: Saving File "<<myFile_;
    dbe_->save(myFile_) ;
  }
}

void SaveDQMFile::endJob(){}

