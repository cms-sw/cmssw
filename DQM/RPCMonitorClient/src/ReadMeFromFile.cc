/*
 *  \author Anna Cimmino
 */
#include <DQM/RPCMonitorClient/interface/ReadMeFromFile.h>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DataFormats
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

// Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// DQM
#include "DQMServices/Core/interface/MonitorElement.h"

#include <map>
#include <sstream>
//#include <math.h>

using namespace edm;
using namespace std;

ReadMeFromFile::ReadMeFromFile(const ParameterSet& ps ){
 
  LogVerbatim ("readFile") << "[ReadMeFromFile]: Constructor";

  myFile_= ps.getUntrackedParameter<string>("InputFile", "uffa.root");
}

ReadMeFromFile::~ReadMeFromFile(){
  dbe_ = 0;
}

void ReadMeFromFile::beginJob(){}

void ReadMeFromFile::beginRun(const Run& r, const EventSetup& iSetup){
  LogVerbatim ("readfile") << "[ReadMeFromFile]: Begin run";
  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(0);
  dbe_->load(myFile_);
}

void ReadMeFromFile::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {}

//called at each event
void ReadMeFromFile::analyze(const edm::Event& iEvent, const edm::EventSetup& c){}


void ReadMeFromFile::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {
 
// //   vector<string>  prova = dbe_->getMEs();
// //   for(unsigned int i=0; i<prova.size(); i++ ){
// //     cout<<prova[i]<<endl;
// //   }

//   cout<<"End lumi block "<<endl;

//   dbe_->setCurrentFolder("RPC/RecHits/Barrel/Wheel_0/sector_1/station_2");
//   //  std::vector<MonitorElement *> mes = dbe_->getAllContents("RPC/RecHits/Barrel/Wheel_0/sector_1/station_2");
//   MonitorElement * me = dbe_->get("RPC/RecHits/Barrel/Wheel_0/sector_1/station_2/Occupancy_W+0_RB2out_S01_Backward");
//  if(me) cout<<"FOUD "<<endl;
// //   if (not mes.empty()) {
// //     std::cout << "found " << mes.size() << " MonitorElements inside 'RPC/RecHits/Barrel/Wheel_0/sector_1/station_2':" << std::endl;
// //     for (size_t i = 0; i < mes.size(); ++i) {
// //       MonitorElement * me = mes[i];
// //       std::cout << '\t' << me->getName() << std::endl;
// //     }
// //   }
}
 
void ReadMeFromFile::endRun(const Run& r, const EventSetup& c){}

void ReadMeFromFile::endJob(){}

