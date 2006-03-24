/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2006/02/21 19:03:12 $
 * $Revision: 1.1 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTDataIntegrityTask.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/DTRawToDigi/interface/DTDataMonitorInterface.h"
#include "EventFilter/DTRawToDigi/interface/DTROS25Data.h"
#include "EventFilter/DTRawToDigi/interface/DTDDUWords.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace edm;

DTDataIntegrityTask::DTDataIntegrityTask(const edm::ParameterSet& ps) {

  cout<<"[DTDataIntegrityTask]: Constructor"<<endl;

  nevents = 0;

  outputFile = ps.getUntrackedParameter<string>("outputFile", "ROS25Test.root");

  parameters = ps;

  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  bookHistos(string("ROS"));

}



DTDataIntegrityTask::~DTDataIntegrityTask() {
 
  cout<<"[DTDataIntegrityTask]: Destructor. Analyzed "<< nevents <<" events"<<endl;
  sleep(10);
  if ( outputFile.size() != 0 ) dbe->save(outputFile);

}


void DTDataIntegrityTask::bookHistos(string folder, int index) {



  // ROS Histograms
  if ( folder == "ROS" ) {
    dbe->setCurrentFolder("DT/ROS25Data/" + folder);

    rosHistos[string("ROSTrailerBits")] = dbe->book1D("ROSTrailerBits","ROSTrailerBits",128,0,128);
    rosHistos[string("ROSError")] = dbe->book2D("ROSError","ROSError",32,0,32,8,0,8);
    rosHistos[string("ROSDebug_BunchNumber")] = 
      dbe->book1D("ROSDebug_BunchNumber","ROSDebug_BunchNumber",2048,0,2048);
    rosHistos[string("ROSDebug_BcntResCntLow")] = 
      dbe->book1D("ROSDebug_BcntResCntLow","ROSDebug_BcntResCntLow",16348,0,16348);
    rosHistos[string("ROSDebug_BcntResCntHigh")] = 
      dbe->book1D("ROSDebug_BcntResCntHigh","ROSDebug_BcntResCntHigh",16348,0,16348);
  }  

  // ROB/TDC Histograms
  if ( folder == "ROB") {
    
    stringstream robID; robID << index;
    dbe->setCurrentFolder("DT/ROS25Data/ROB/" + folder + robID.str());
    string histoName = folder + robID.str() + "_Occupancy";
    (robHistos[histoName])[index] = dbe->book2D(histoName,histoName,32,0,32,4,0,4);
    histoName = folder + robID.str() + "_TimeBox";
    (robHistos[histoName])[index] = dbe->book1D(histoName,histoName,
						(parameters.getUntrackedParameter<int>("timeBoxUpperBound",10000)-
						parameters.getUntrackedParameter<int>("timeBoxLowerBound",0))/2,
						parameters.getUntrackedParameter<int>("timeBoxLowerBound",0),
						parameters.getUntrackedParameter<int>("timeBoxUpperBound",10000));

  }

  cout<<"[DTDataIntegrityTask]: Histograms booked "<<endl;
}



void DTDataIntegrityTask::process(DTROS25Data & data) {
  
  nevents++;
  if (nevents%100 == 0) 
    cout<<"[DTDataIntegrityTask]: "<<nevents<<" events analyzed"<<endl;
  
  for (vector<DTROSTrailerWord>::const_iterator trailer_it = data.getROSTrailers().begin();
       trailer_it != data.getROSTrailers().end(); trailer_it++) {
    int datum = (*trailer_it).TFF() << (23-16) 
      | (*trailer_it).TPX() << (22-16) 
      | (*trailer_it).ECHO() << (20-16) 
      | (*trailer_it).ECLO() << (18-16) 
      | (*trailer_it).BCO()<< (16-16);
    (rosHistos.find(string("ROSTrailerBits"))->second)->Fill(datum);
  }

  for (vector<DTROSErrorWord>::const_iterator error_it = data.getROSErrors().begin();
       error_it != data.getROSErrors().end(); error_it++) {
    (rosHistos.find(string("ROSError"))->second)->Fill((*error_it).robID(), (*error_it).errorType());
  }
  
  for (vector<DTROSDebugWord>::const_iterator debug_it = data.getROSDebugs().begin();
       debug_it != data.getROSDebugs().end(); debug_it++) {
    (rosHistos.find(string("ROSDebug_BunchNumber"))->second)->Fill((*debug_it).debugMessage());
    (rosHistos.find(string("ROSDebug_BcntResCntLow"))->second)->Fill((*debug_it).debugMessage());
    (rosHistos.find(string("ROSDebug_BcntResCntHigh"))->second)->Fill((*debug_it).debugMessage());
  }
  
  
  for (vector<DTTDCData>::const_iterator tdc_it = data.getTDCData().begin();
       tdc_it != data.getTDCData().end(); tdc_it++) {

    stringstream robID; robID << (*tdc_it).first;
    DTTDCMeasurementWord tdcDatum = (*tdc_it).second;

    string histoName = "ROB" + robID.str() + "_Occupancy";
    if (robHistos[histoName].find((*tdc_it).first) != robHistos[histoName].end())
      (robHistos.find(histoName)->second).find((*tdc_it).first)->second->Fill(tdcDatum.tdcChannel(),tdcDatum.tdcID());
    else {
      bookHistos( string("ROB"), (*tdc_it).first);
      (robHistos.find(histoName)->second).find((*tdc_it).first)->second->Fill(tdcDatum.tdcChannel(),tdcDatum.tdcID());
    }

    histoName = "ROB" + robID.str() + "_TimeBox";
    if (robHistos[histoName].find((*tdc_it).first) != robHistos[histoName].end())
      (robHistos.find(histoName)->second).find((*tdc_it).first)->second->Fill(tdcDatum.tdcTime());
    else {
      bookHistos( string("ROB"), (*tdc_it).first);
      (robHistos.find(histoName)->second).find((*tdc_it).first)->second->Fill(tdcDatum.tdcTime());
    }


  }

}


