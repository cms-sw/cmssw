/*
 * \file DTLocalRecoTask.cc
 * 
 * $Date: 2006/07/03 15:30:24 $
 * $Revision: 1.6 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include "DQM/DTMonitorModule/interface/DTLocalRecoTask.h"
#include "DQM/DTMonitorModule/src/DTSegmentAnalysis.h"
#include "DQM/DTMonitorModule/src/DTResolutionAnalysis.h"

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
							     theSegmentAnalysis(0),
							     theResolutionAnalysis(0) {
  debug = pset.getUntrackedParameter<bool>("debug", "false");

  if(debug)
    cout << "[DTLocalRecoTask] Constructor called!" << endl;

  // Get the DQM needed services
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  dbe->setVerbose(1);


  edm::Service<MonitorDaemon>().operator->(); 	 

  theRootFileName = pset.getUntrackedParameter<string>("rootFileName", "DTLocalRecoTask.root");

  writeHisto = pset.getUntrackedParameter<bool>("writeHisto", true);
  
  if (dbe)
    dbe->setCurrentFolder("DT/DTLocalRecoTask");
  
  doSegmentAnalysis = pset.getUntrackedParameter<bool>("doSegmentAnalysis", true);
  doResolutionAnalysis = pset.getUntrackedParameter<bool>("doResolutionAnalysis", true);
  
  // Create the classes which really make the analysis
  if(doSegmentAnalysis)
    theSegmentAnalysis =
      new DTSegmentAnalysis(pset.getParameter<ParameterSet>("segmentAnalysisConfig"), dbe);
  
  if(doResolutionAnalysis)
    theResolutionAnalysis =
      new DTResolutionAnalysis(pset.getParameter<ParameterSet>("resolutionAnalysisConfig"), dbe);


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
  if ( writeHisto ) 
    dbe->save(theRootFileName);
  dbe->setCurrentFolder("DT/DTLocalRecoTask");
  dbe->removeContents();
}

void DTLocalRecoTask::analyze(const Event& event, const EventSetup& setup){
  if(doSegmentAnalysis)
    theSegmentAnalysis->analyze(event, setup);
  if(doResolutionAnalysis)
    theResolutionAnalysis->analyze(event, setup);

}

