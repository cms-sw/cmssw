/*
 * \file DQMAnalyzer.cc
 * 
 * $Date: 2007/04/03 09:51:57 $
 * $Revision: 1.3 $
 * \author M. Zanetti - CERN PH
 *
 */

#include "DQMServices/Components/interface/DQMAnalyzer.h"

// Framework
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>


#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

DQMAnalyzer::DQMAnalyzer(const edm::ParameterSet& ps):
irun_(0),ilumisec_(0),ievent_(0),itime_(0),
nevt_(0),nlumisecs_(0),
saved_(false),
debug_(false)
//,actonLS_(false)
{
  
  parameters = ps;

  cout<<"DQMAnalyzer::constructor"<<endl;
  
  // get backendinterface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
    cout << " DaqMonitorBEInterface " << endl;

  // some other environment parms (FIXME: put into SetParameters() function)
  if ( parameters.getUntrackedParameter<bool>("enableMonitorDaemon", true) ) {
    Service<MonitorDaemon> daemon;
    daemon.operator->();
    cout << " MonitorDaemon enabled " << endl;
  } 
  else {
    cout<<"DQMAnalyzer: Warning, MonitorDaemon service not enabled"<<endl;
  }

  // set parameters   
  PSprescale = parameters.getUntrackedParameter<int>("diagnosticPrescale", 1);
    cout << " PSprescale = " << PSprescale << endl;


// FIXME make client act upon event counter or LS or collector update ...
//  PSprescaleLS = parameters.getUntrackedParameter<int>("PrescaleLS", -1);
//    cout << " PSprescaleLS  = " << PSprescaleLS << endl;
//  if (PSprescaleLS>0) { 
//    actonLS_=true;
//    cout << " Note: This module acts on every " << PSprescaleLS << 
//            " endLuminosityBlocks " << endl;
//  }
    
  PSrootFolder = parameters.getUntrackedParameter<string>("folderRoot", "");

  if (PSrootFolder.size() != 0) {
    if( PSrootFolder.substr(PSrootFolder.size()-1, 1) != "/" ) 
    PSrootFolder = PSrootFolder + "/";
    cout << " PSrootFolder  = " << PSrootFolder << endl;
  }

}

DQMAnalyzer::~DQMAnalyzer(){

  if (debug_) cout<<"DQMAnalyzer::destructor"<<endl;

}

void DQMAnalyzer::beginJob(const edm::EventSetup& c){
  
  nevt_=0;
  nlumisecs_=0;
  
  // book framework ME
  dbe->setVerbose(1);
  dbe->setCurrentFolder(parameters.getUntrackedParameter<string>("eventInfoFolder", "EventInfo/")) ;
  runId_ = dbe->bookInt("iRun");
  lumisecId_ = dbe->bookInt("iLumiSection");
  eventId_ = dbe->bookInt("iEvent");
  timeStamp_ = dbe->bookFloat("timeStamp");

}

void DQMAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& c){
 
  if (debug_ || nevt_==1) cout<<"DQMAnalyzer::analyze"<<endl;

  // set counters and flags
  saved_ = false;
  nevt_++;

  // environment datamembers
  irun_     = e.id().run();
  ilumisec_ = e.luminosityBlock();
  ievent_   = e.id().event();
  itime_    = e.time().value();

  if (debug_) cout << nevt_ << " " << irun_ << " " << ilumisec_ << " " << ievent_ << " " << itime_ << endl; 
  
  // ME
  if (runId_)     runId_->Fill(irun_);
  if (lumisecId_) lumisecId_->Fill(ilumisec_); 
  if (eventId_)   eventId_->Fill(ievent_);
  if (timeStamp_) timeStamp_->Fill(itime_); 

}

void DQMAnalyzer::beginRun(const edm::EventSetup& c){
   cout <<"DQMAnalyzer::beginRun"<<endl;
   cout <<"FIXME reset histos here"<<endl;
}
void DQMAnalyzer::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){
   if (debug_ || nlumisecs_==0) cout <<"DQMAnalyzer::beginLuminosityBlock"<<endl;
   nlumisecs_++;
   cout <<"nlumisecs_: "<<nlumisecs_<<endl;
}
void DQMAnalyzer::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){
   if (debug_ || nlumisecs_==1) cout <<"DQMAnalyzer::endLuminosityBlock"<<endl;
}
void DQMAnalyzer::endRun(const edm::Run& run, const edm::EventSetup& c){
   cout <<"DQMAnalyzer::endRun"<<endl;
   save();
}
void DQMAnalyzer::endJob() { 
   cout <<"DQMAnalyzer::endJob"<<endl;
   save();
}

void DQMAnalyzer::save(std::string flag){

  if (debug_) cout <<"DQMAnalyzer::save"<<endl;
  if (saved_) return; // save only once per event
  std::string name = parameters.getUntrackedParameter<string>("outputFile","") ;
  
  // default name
  if (name=="") return;
  
  if (debug_) cout <<"DQMAnalyzer::save: saving"<<endl;


  // temporarily strip off prefix DQM and extensions .root
  if( name.size() != 0) {
    for( unsigned int i = 0; i < name.size(); i++ ) {
      if( name.substr(i, 5) == ".root" ) name.replace(i, 5, "");
      if( name.substr(i, 4) == "DQM_" ) name.replace(i, 4, "");
      if( name.substr(i, 3) == "DQM" ) name.replace(i, 3, "");
    }
  }

  // add runnumber  
  char run[10];
  if(irun_>0) sprintf(run,"%09d", irun_);
  else sprintf(run,"%09d", 0);

  if (flag=="endRun") {
    string saver = "DQM_"+name+"_"+run+".root";
    dbe->save(saver);
    saved_=true; // save only once per event
    return;
  }
  
  // add lumisection number  
  char lumisec[10];
  if(ilumisec_>0) sprintf(lumisec,"%06d", ilumisec_);
  else sprintf(lumisec,"%06d", 0);
  
  string saver = "DQM_"+name+"_"+run+"_"+lumisec+".root";
  dbe->save(saver);
  saved_=true; // save only once per event

}

