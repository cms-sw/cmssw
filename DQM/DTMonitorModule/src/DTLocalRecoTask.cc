/*
 * \file DTLocalRecoTask.cc
 * 
 * $Date: 2006/06/01 11:09:27 $
 * $Revision: 1.2 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "DQM/DTMonitorModule/interface/DTLocalRecoTask.h"
#include "DQM/DTMonitorModule/src/DTSegmentAnalysis.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

// #include <DataFormats/DTDigi/interface/DTDigi.h>
// #include <DataFormats/DTDigi/interface/DTDigiCollection.h>
// #include <DataFormats/MuonDetId/interface/DTLayerId.h>

#include <iostream>

using namespace std;
using namespace edm;


DTLocalRecoTask::DTLocalRecoTask(const ParameterSet& pset) : dbe(0),
							     theSegmentAnalysis(0) {
  debug = pset.getUntrackedParameter<bool>("debug", "false");

  if(debug)
    cout << "[DTLocalRecoTask] Constructor called!" << endl;

  // Get the DQM needed services
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);


  edm::Service<MonitorDaemon>().operator->(); 	 

  theRootFileName = pset.getUntrackedParameter<string>("rootFileName", "DTLocalRecoTask.root");

  
  if (dbe)
    dbe->setCurrentFolder("DT/DTLocalRecoTask");


  
  // Create the classes which really make the analysis
  theSegmentAnalysis = new DTSegmentAnalysis(pset.getParameter<ParameterSet>("SegmentAnalysisConfig"), dbe);
  
  


//   logFile.open("DTLocalRecoTask.log");





}

DTLocalRecoTask::~DTLocalRecoTask(){
  if(debug)
    cout << "[DTLocalRecoTask] Destructor called!" << endl;
  //   logFile.close();

}

void DTLocalRecoTask::beginJob(const EventSetup& setup){

  //dbe->


}

void DTLocalRecoTask::endJob(){
  // Write the histos
  dbe->save(theRootFileName);

}

void DTLocalRecoTask::analyze(const Event& event, const EventSetup& setup){
 
  theSegmentAnalysis->analyze(event, setup);
 

}

