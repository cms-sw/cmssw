/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/11/05 11:39:46 $
 *  $Revision: 1.8 $
 *  \author C. Battilana S. Marcellini - INFN Bologna
 */


// This class header
#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"

// Framework headers
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

// Root
#include "TF1.h"
#include "TProfile.h"


//C++ headers
#include <iostream>
#include <sstream>

using namespace edm;
using namespace std;


DTLocalTriggerBaseTest::~DTLocalTriggerBaseTest(){

  LogVerbatim(category()) << "[" << testName << "Test]: analyzed " << nevents << " events";

}

void DTLocalTriggerBaseTest::beginJob(const edm::EventSetup& context){

  LogVerbatim(category()) << "[" << testName << "Test]: BeginJob";
  nevents = 0;
  nLumiSegs = 0;
  context.get<MuonGeometryRecord>().get(muonGeom);
  
}


void DTLocalTriggerBaseTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  LogTrace(category()) <<"[" << testName << "Test]: Begin of LS transition";

  // Get the run number
  run = lumiSeg.run();

}


void DTLocalTriggerBaseTest::analyze(const edm::Event& e, const edm::EventSetup& context){

  nevents++;
  LogTrace(category()) << "[" << testName << "Test]: "<<nevents<<" events";

}


void DTLocalTriggerBaseTest::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) {
  
  if (!runOnline) return;

  LogVerbatim("DTDQM|DTMonitorClient|DTLocalTriggerTest") <<"[" << testName << "Test]: End of LS transition, performing the DQM client operation";

  // counts number of lumiSegs and prescale
  nLumiSegs++;
  if ( nLumiSegs%prescaleFactor != 0 ) return;

  LogVerbatim("DTDQM|DTMonitorClient|DTLocalTriggerTest") <<"[" << testName << "Test]: "<<nLumiSegs<<" updates";  
  runClientDiagnostic();

}


void DTLocalTriggerBaseTest::endJob(){
  
  LogTrace(category()) << "[" << testName << "Test] endjob called!";

  if (!runOnline) {
    LogVerbatim(category()) << "[" << testName << "Test] Client called in offline mode, performing client operations";

    if (dbe->dirExists(topFolder())) {
      //dbe->showDirStructure();
      runClientDiagnostic();
    }
    else {
      LogVerbatim(category()) << "[" << testName << "Test] trigger ME dir does not exist! Skipping";  
    }

  }

}


void DTLocalTriggerBaseTest::setConfig(const edm::ParameterSet& ps, string name){

  testName=name;

  LogTrace(category()) << "[" << testName << "Test]: Constructor";

  sourceFolder = ps.getUntrackedParameter<string>("folderRoot", ""); 
  runOnline = ps.getUntrackedParameter<bool>("runOnline",true);
  hwSources = ps.getUntrackedParameter<vector<string> >("hwSources");

  if (ps.getUntrackedParameter<bool>("localrun",true)) {
    trigSources.push_back("");
  }
  else {
    trigSources = ps.getUntrackedParameter<vector<string> >("trigSources");
  }

  parameters = ps;
  nevents = 0;
  dbe = edm::Service<DQMStore>().operator->();

  prescaleFactor = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);

}


string DTLocalTriggerBaseTest::fullName (string htype) {

  return hwSource + "_" + htype + trigSource;

}

string DTLocalTriggerBaseTest::getMEName(string histoTag, string subfolder, const DTChamberId & chambid) {

  stringstream wheel; wheel << chambid.wheel();
  stringstream station; station << chambid.station();
  stringstream sector; sector << chambid.sector();

  string hwFolder = hwSource=="DCC" ? "DCC/" : ""; 

  string folderName = 
    topFolder() + hwFolder + "Wheel" +  wheel.str() +
    "/Sector" + sector.str() +
    "/Station" + station.str() + "/" + subfolder + "/";  

  string histoname = sourceFolder + folderName 
    + fullName(histoTag) 
    + "_W" + wheel.str()  
    + "_Sec" + sector.str()
    + "_St" + station.str();
  
  return histoname;
  
}


// void DTLocalTriggerBaseTest::setLabelPh(MonitorElement* me){

//   for (int i=0; i<48; ++i){
//     stringstream label;
//     int stat = (i%4) +1;
//     if (stat==1) label << "Sec " << i/4 +1 << " ";
//     me->setBinLabel(i+1,label.str().c_str());
//   }

// }

// void DTLocalTriggerBaseTest::setLabelTh(MonitorElement* me){

//   for (int i=0; i<36; ++i){
//     stringstream label;
//     int stat = (i%3) +1;
//     if (stat==1) label << "Sec " << i/3 +1 << " ";
//     me->setBinLabel(i+1,label.str().c_str());
//   }

// }


void DTLocalTriggerBaseTest::bookSectorHistos(int wheel,int sector,string folder, string hTag) {
  
  stringstream wh; wh << wheel;
  stringstream sc; sc << sector;
  int sectorid = (wheel+3) + (sector-1)*5;
  string hwFolder = hwSource=="DCC" ? "DCC/" : "";
  string basedir = topFolder()+hwFolder+"Wheel"+wh.str()+"/Sector"+sc.str()+"/";
  if (folder!="") {
    basedir += folder +"/";
  }
  dbe->setCurrentFolder(basedir);

  string fullTag = fullName(hTag);
  string hname    = fullTag + "_W" + wh.str()+"_Sec" +sc.str();
  LogTrace(category()) << "[" << testName << "Test]: booking " << basedir << hname;
  if (hTag.find("Phi") != string::npos || 
      hTag.find("TkvsTrig") != string::npos ){    
    MonitorElement* me = dbe->book1D(hname.c_str(),hname.c_str(),4,0.5,4.5);
    me->setBinLabel(1,"MB1",1);
    me->setBinLabel(2,"MB2",1);
    me->setBinLabel(3,"MB3",1);
    me->setBinLabel(4,"MB4",1);
    secME[sectorid][fullTag] = me;
    return;
  }
  
  if (hTag.find("Theta") != string::npos){
    MonitorElement* me =dbe->book1D(hname.c_str(),hname.c_str(),3,0.5,3.5);
    me->setBinLabel(1,"MB1",1);
    me->setBinLabel(2,"MB2",1);
    me->setBinLabel(3,"MB3",1);
    secME[sectorid][fullTag] = me;
    return;
  }
  
}

void DTLocalTriggerBaseTest::bookCmsHistos( string hTag ) {

  string basedir = topFolder();
   if(hwSource == "DCC") 
      basedir += hwSource + "/";
  dbe->setCurrentFolder(basedir);

  string hname = fullName(hTag);
  LogTrace(category()) << "[" << testName << "Test]: booking " << basedir << hname;


    MonitorElement* me = dbe->book2D(hname.c_str(),hname.c_str(),12,1,13,5,-2,3);
    me->setAxisTitle("Sector",1);
    me->setAxisTitle("Wheel",2);
    cmsME[hname] = me;

}

void DTLocalTriggerBaseTest::bookWheelHistos(int wheel, string folder, string hTag) {
  
  stringstream wh; wh << wheel;
  string basedir;  
  string hwFolder = hwSource=="DCC" ? "DCC/" : "" ;  
  if (hTag.find("Summary") != string::npos ) {
    basedir = topFolder() + hwFolder;   //Book summary histo outside wheel directories
  } else {
    basedir = topFolder() + hwFolder + "Wheel" + wh.str() + "/" ;
    if (folder != "") {
      basedir += folder +"/" ;
    }
  }
  dbe->setCurrentFolder(basedir);

  string fullTag = fullName(hTag);
  string hname    = fullTag+ "_W" + wh.str();

  LogTrace(category()) << "[" << testName << "Test]: booking "<< basedir << hname;
  
  if (hTag.find("Phi")!= string::npos ||
      hTag.find("Summary") != string::npos ){    
    MonitorElement* me = dbe->book2D(hname.c_str(),hname.c_str(),12,1,13,4,1,5);

//     setLabelPh(me);
    me->setBinLabel(1,"MB1",2);
    me->setBinLabel(2,"MB2",2);
    me->setBinLabel(3,"MB3",2);
    me->setBinLabel(4,"MB4",2);
    me->setAxisTitle("Sector",1);
    
    whME[wheel][fullTag] = me;
    return;
  }
  
  if (hTag.find("Theta") != string::npos){
    MonitorElement* me =dbe->book2D(hname.c_str(),hname.c_str(),12,1,13,3,1,4);

//     setLabelTh(me);
    me->setBinLabel(1,"MB1",2);
    me->setBinLabel(2,"MB2",2);
    me->setBinLabel(3,"MB3",2);
    me->setAxisTitle("Sector",1);

    whME[wheel][fullTag] = me;
    return;
  }
  
}

pair<float,float> DTLocalTriggerBaseTest::phiRange(const DTChamberId& id){

  float min,max;
  int station = id.station();
  int sector  = id.sector(); 
  int wheel   = id.wheel();
  
  const DTLayer  *layer = muonGeom->layer(DTLayerId(id,1,1));
  DTTopology topo = layer->specificTopology();
  min = topo.wirePosition(topo.firstChannel());
  max = topo.wirePosition(topo.lastChannel());

  if (station == 4){
    
    const DTLayer *layer2;
    float lposx;
    
    if (sector == 4){
      layer2  = muonGeom->layer(DTLayerId(wheel,station,13,1,1));
      lposx = layer->toLocal(layer2->position()).x();
    }
    else if (sector == 10){
      layer2 = muonGeom->layer(DTLayerId(wheel,station,14,1,1));
      lposx = layer->toLocal(layer2->position()).x();
    }
    else
      return make_pair(min,max);
    
    DTTopology topo2 = layer2->specificTopology();

    if (lposx>0){
      max = lposx*.5+topo2.wirePosition(topo2.lastChannel());
      min -= lposx*.5;
    }
    else{
      min = lposx*.5+topo2.wirePosition(topo2.firstChannel());
      max -= lposx*.5;
    }
      
  }
  
  return make_pair(min,max);

}
