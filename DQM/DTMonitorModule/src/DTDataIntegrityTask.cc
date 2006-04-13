/*
 * \file DTDigiTask.cc
 * 
 * $Date: 2006/04/10 12:30:06 $
 * $Revision: 1.3 $
 * \author M. Zanetti - INFN Padova
 *
*/

#include <DQM/DTMonitorModule/interface/DTDataIntegrityTask.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/DTRawToDigi/interface/DTDataMonitorInterface.h"
#include "EventFilter/DTRawToDigi/interface/DTControlData.h"
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

}



DTDataIntegrityTask::~DTDataIntegrityTask() {
 
  cout<<"[DTDataIntegrityTask]: Destructor. Analyzed "<< nevents <<" events"<<endl;
  sleep(10);
  if ( outputFile.size() != 0 ) dbe->save(outputFile);

}

/*
Folder Structure:
- One folder for each DDU, named FEDn
- Inside each DDU folder the DDU histos and the ROSn folder
- Inside each ROS folder the ROS histos and the ROBn folder
- Inside each ROB folder one occupancy plot and the TimeBoxes
  with the chosen granularity (simply change the histo name)
*/


void DTDataIntegrityTask::bookHistos(string folder, DTROChainCoding code) {

  stringstream dduID_s; dduID_s << code.getDDU();
  stringstream rosID_s; rosID_s << code.getROS();
  stringstream robID_s; robID_s << code.getROB();
  string histoType;
  string histoName;

  // DDU Histograms
  if ( folder == "DDU" ) {
    dbe->setCurrentFolder("DT/FED" + dduID_s.str());
  }

  // ROS Histograms
  if ( folder == "ROS" ) {

    dbe->setCurrentFolder("DT/FED" + dduID_s.str() + "/" + folder + rosID_s.str());

    histoType = "ROSTrailerBits";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSTrailerBits";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,128,0,128);

    histoType = "ROSError";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSError";
    (rosHistos[histoType])[code.getROSID()] = dbe->book2D(histoName,histoName,32,0,32,8,0,8);

    histoType = "ROSDebug_BunchNumber";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSDebug_BunchNumber";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,2048,0,2048);

    histoType = "ROSDebug_BcntResCntLow";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSDebug_BcntResCntLow";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,16348,0,16348);

    histoType = "ROSDebug_BcntResCntHigh";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSDebug_BcntResCntHigh";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,16348,0,16348);
  }  

  // ROB/TDC Histograms
  if ( folder == "ROB_O") {
    
    dbe->setCurrentFolder("DT/FED" + dduID_s.str()+"/ROS"+rosID_s.str()+"/ROB"+robID_s.str());

    histoType = "Occupancy";
    histoName = "FED" + dduID_s.str() + "_ROS" + rosID_s.str() + "_ROB"+robID_s.str()+"_Occupancy";
    (robHistos[histoType])[code.getROBID()] = dbe->book2D(histoName,histoName,32,0,32,4,0,4);

  }

  if ( folder == "ROB_T") {

    dbe->setCurrentFolder("DT/FED" + dduID_s.str()+"/ROS"+rosID_s.str()+"/ROB"+robID_s.str());

    histoType = "TimeBox";
    histoName = "FED" + dduID_s.str() + "_ROS" + rosID_s.str() + "_ROB" + robID_s.str()+"_TimeBox";
    int index;
    switch (parameters.getUntrackedParameter<int>("TBhistoGranularity",1)) {
    case 1: // ROB
      index = code.getROBID();
      break;
    case 2: // TDC
      index = code.getTDCID();
      break;
    case 3: // Ch
      index = code.getChannelID();
      break;
    default: // ROB
      index = code.getROBID();      
    }
    (robHistos[histoType])[index] = dbe->book1D(histoName,histoName,
						(parameters.getUntrackedParameter<int>("timeBoxUpperBound",10000)-
						 parameters.getUntrackedParameter<int>("timeBoxLowerBound",0))/2,
						parameters.getUntrackedParameter<int>("timeBoxLowerBound",0),
						parameters.getUntrackedParameter<int>("timeBoxUpperBound",10000));
    
  }
  
}



void DTDataIntegrityTask::processROS25(DTROS25Data & data, int ddu, int ros) {
  
  nevents++;
  if (nevents%1000 == 0) 
    cout<<"[DTDataIntegrityTask]: "<<nevents<<" events analyzed"<<endl;
  
  DTROChainCoding code;
  code.setDDU(ddu);
  code.setROS(ros);

  string histoType;

  /// ROS Data
  histoType = "ROSTrailerBits";
  int datum = 
    data.getROSTrailer().TFF() << (23-16) | 
    data.getROSTrailer().TPX() << (22-16) |
    data.getROSTrailer().ECHO() << (20-16) |
    data.getROSTrailer().ECLO() << (18-16) |
    data.getROSTrailer().BCO()<< (16-16);
  if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()) {
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(datum);
  }
  else {
    bookHistos( string("ROS"), code);
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(datum);
  }
  
  histoType = "ROSError";
  for (vector<DTROSErrorWord>::const_iterator error_it = data.getROSErrors().begin();
       error_it != data.getROSErrors().end(); error_it++) {
    if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end())
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*error_it).robID(), 
									      (*error_it).errorType());
    else {
      bookHistos( string("ROS"), code);
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*error_it).robID(), 
									      (*error_it).errorType());
    }
  }
  
  for (vector<DTROSDebugWord>::const_iterator debug_it = data.getROSDebugs().begin();
       debug_it != data.getROSDebugs().end(); debug_it++) {
    histoType = "ROSDebug_BunchNumber";
    if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end())
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*debug_it).debugMessage());
    else {
      bookHistos( string("ROS"), code);
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*debug_it).debugMessage());
    }

    histoType = "ROSDebug_BcntResCntLow";
    if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end())
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*debug_it).debugMessage());
    else {
      bookHistos( string("ROS"), code);
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*debug_it).debugMessage());
    }

    histoType = "ROSDebug_BcntResCntHigh";
    if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end())
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*debug_it).debugMessage());
    else {
      bookHistos( string("ROS"), code);
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*debug_it).debugMessage());
    }
  }
  
  /// TDC Data  
  for (vector<DTTDCData>::const_iterator tdc_it = data.getTDCData().begin();
       tdc_it != data.getTDCData().end(); tdc_it++) {


    stringstream robID_s; robID_s << (*tdc_it).first;
    DTTDCMeasurementWord tdcDatum = (*tdc_it).second;
    int index;
    switch (parameters.getUntrackedParameter<int>("TBhistoGranularity",1)) {
    case 1:
      code.setROB((*tdc_it).first);
      index = code.getROBID();
      break;
    case 2:
      code.setTDC(tdcDatum.tdcID());
      index = code.getTDCID();
      break;
    case 3:
      code.setChannel(tdcDatum.tdcChannel());
      index = code.getChannelID();
      break;
    default:
      code.setROB((*tdc_it).first);
      index = code.getROBID();
    }


    histoType = "Occupancy";
    if (robHistos[histoType].find(code.getROBID()) != robHistos[histoType].end()) {
      (robHistos.find(histoType)->second).find(code.getROBID())->second->Fill(tdcDatum.tdcChannel(),
									      tdcDatum.tdcID());
    }
    else {
      bookHistos( string("ROB_O"), code);
      (robHistos.find(histoType)->second).find(code.getROBID())->second->Fill(tdcDatum.tdcChannel(),
									      tdcDatum.tdcID());
    }

    histoType = "TimeBox";
    if (robHistos[histoType].find(index) != robHistos[histoType].end()) {
      (robHistos.find(histoType)->second).find(index)->second->Fill(tdcDatum.tdcTime());

    }
    else {
      bookHistos( string("ROB_T"), code);
      (robHistos.find(histoType)->second).find(index)->second->Fill(tdcDatum.tdcTime());
    }
  }

}

void DTDataIntegrityTask::processFED(DTDDUData & data, int ddu) {

}
