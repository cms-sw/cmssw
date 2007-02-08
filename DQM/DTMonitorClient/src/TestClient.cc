/*
 * \file DTNoiseClient.cc
 * 
 * $Date: 2006/08/10 16:27:18 $
 * $Revision: 1.5 $
 * \author S. Bolognesi - M. Zanetti
 *
 */


#include <DQM/DTMonitorClient/interface/TestClient.h>

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

TestClient::TestClient(const edm::ParameterSet& ps){
  
  debug = ps.getUntrackedParameter<bool>("debug", "false");
  if(debug)
    cout<<"[TestClient]: Constructor"<<endl;

  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);

//   edm::Service<MonitorDaemon> daemon; 	 
//   daemon.operator->();

}


TestClient::~TestClient(){

  if(debug)
    cout << "TestClient: analyzed " << nevents << " events" << endl;

}

void TestClient::endJob(){

  if(debug)
    cout<<"[TestClient] endjob called!"<<endl;

  if ( parameters.getUntrackedParameter<bool>("writeHisto", true) ) 
    dbe->save(parameters.getUntrackedParameter<string>("outputFile", "TestClient.root"));
  
  dbe->rmdir("DT/TestClient");
}

void TestClient::beginJob(const edm::EventSetup& context){

  if(debug)
    cout<<"[TestClient]: BeginJob"<<endl;

  nevents = 0;

  // Get the geometry
  context.get<MuonGeometryRecord>().get(muonGeom);

}


void TestClient::analyze(const edm::Event& e, const edm::EventSetup& context){

  context.get<DTTtrigRcd>().get(tTrigMap);

  nevents++;
  if (nevents%1 == 0 && debug) 
    cout<<"[TestClient]: "<<nevents<<" events analyzed"<<endl;

  
}



