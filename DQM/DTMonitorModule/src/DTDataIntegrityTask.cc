
/*
 * \file DTDataIntegrityTask.cc
 * 
 * $Date: 2008/11/06 16:01:13 $
 * $Revision: 1.50 $
 * \author M. Zanetti (INFN Padova), S. Bolognesi (INFN Torino)
 *
 */

#include <DQM/DTMonitorModule/interface/DTDataIntegrityTask.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/DTRawToDigi/interface/DTDataMonitorInterface.h"
#include "EventFilter/DTRawToDigi/interface/DTControlData.h"
#include "EventFilter/DTRawToDigi/interface/DTDDUWords.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include <math.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace edm;
int FirstRos=0,nevents=0,n,m;
const unsigned long long max_bx = 59793997824ULL;
#include "ROSDebugUtility.h"

DTDataIntegrityTask::DTDataIntegrityTask(const edm::ParameterSet& ps,edm::ActivityRegistry& reg) : dbe(0) {

  // Register the methods that we want to schedule
  //   reg.watchPostEndJob(this,&DTDataIntegrityTask::postEndJob);
  reg.watchPostBeginJob(this,&DTDataIntegrityTask::postBeginJob);
  reg.watchPreProcessEvent(this,&DTDataIntegrityTask::preProcessEvent);
  
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: Constructor" <<endl;

  neventsDDU = 0;
  neventsROS25 = 0;

//   //If you want info VS time histos
//   doTimeHisto =  ps.getUntrackedParameter<bool>("doTimeHisto", false);
  // Plot quantities about SC
  getSCInfo = ps.getUntrackedParameter<bool>("getSCInfo", false);

  // flag to toggle the creation of only the summaries (for HLT running)
  hltMode = ps.getUntrackedParameter<bool>("hltMode", false);
}



DTDataIntegrityTask::~DTDataIntegrityTask() {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    <<"[DTDataIntegrityTask]: Destructor. Analyzed "<< neventsDDU <<" events"<<endl;
}



/*
  Folder Structure:
  - One folder for each DDU, named FEDn
  - Inside each DDU folder the DDU histos and the ROSn folder
  - Inside each ROS folder the ROS histos and the ROBn folder
  - Inside each ROB folder one occupancy plot and the TimeBoxes
  with the chosen granularity (simply change the histo name)
*/

void DTDataIntegrityTask::postEndJob(){
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << "[DTDataIntegrityTask]: postEndJob called!" <<endl;

//   if(doTimeHisto) TimeHistos("Event_word_vs_time");	
	
}


void DTDataIntegrityTask::bookHistos() {
  // Standard FED integrity histos
  if(!hltMode) dbe->setCurrentFolder("DT/FEDIntegrity_SM/");
  else dbe->setCurrentFolder("DT/FEDIntegrity/");

  hFEDEntry = dbe->book1D("FEDEntries","# entries per DT FED",5,770,775);
  hFEDFatal = dbe->book1D("FEDFatal","# fatal errors DT FED",5,770,775);
  hFEDNonFatal = dbe->book1D("FEDNonFatal","# NON fatal errors DT FED",5,770,775);

}



void DTDataIntegrityTask::bookHistos(string folder, DTROChainCoding code) {

  stringstream dduID_s; dduID_s << code.getDDU();
  stringstream rosID_s; rosID_s << code.getROS();
  stringstream robID_s; robID_s << code.getROB();
  int wheel = code.getDDUID() - 772;
  stringstream wheel_s; wheel_s << wheel;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << " Booking histos for FED: " << code.getDDU() << " ROS: " << code.getROS()
    << " ROB: " << code.getROB() << " folder: " << folder << endl;

  string histoType;
  string histoName;

  // DDU Histograms
  if (folder == "DDU") {
    dbe->setCurrentFolder(topFolder() + "FED" + dduID_s.str());

    histoType = "TTSValues";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoName,7,0,7);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"disconnected",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"warning overflow",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"out of synch",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(4,"busy",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(5,"ready",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(6,"error",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(7,"disconnected",1);	

    histoType = "TTS_2";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoName,21,0,21);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"L1A mismatch",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"BX mismatch",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"L1A Full ch1-4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(4,"L1A Full ch5-8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(5,"L1A Full ch9-12",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(6,"Input Full ch1-4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(7,"Input Full ch5-8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(8,"Input Full ch9-12",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(9,"Output FIFO Full",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(10,"error ROS 1",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(11,"error ROS 2",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(12,"error ROS 3",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(13,"error ROS 4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(14,"error ROS 5",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(15,"error ROS 6",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(16,"error ROS 7",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(17,"error ROS 8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(18,"error ROS 9",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(19,"error ROS 10",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(20,"error ROS 11",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(21,"error ROS 12",1);

    histoType = "TTS_12";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoName,21,0,21);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"L1A mismatch",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"BX mismatch",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"L1A Full ch1-4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(4,"L1A Full ch5-8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(5,"L1A Full ch9-12",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(6,"Input Full ch1-4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(7,"Input Full ch5-8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(8,"Input Full ch9-12",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(9,"Output FIFO Full",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(10,"error ROS 1",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(11,"error ROS 2",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(12,"error ROS 3",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(13,"error ROS 4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(14,"error ROS 5",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(15,"error ROS 6",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(16,"error ROS 7",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(17,"error ROS 8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(18,"error ROS 9",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(19,"error ROS 10",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(20,"error ROS 11",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(21,"error ROS 12",1);


    histoType = "EventLenght";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    string histoTitle = "Event Lenght (Bytes) FED " +  dduID_s.str();
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoTitle,1000,0,1000);
 
    histoType = "EventType";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoName,7,1,8);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"physics",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"calibration",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"test",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(4,"technical",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(5,"simulated",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(6,"traced",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(7,"error",1);	
  
    histoType = "ROSList";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoName,13,0,13);
    
    histoType = "ROSStatus";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = dbe->book2D(histoName,histoName,9,0,9,12,0,12);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"ch.enabled",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"timeout",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"ev.trailer lost",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(4,"opt.fiber lost",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(5,"tlk.prop.error",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(6,"tlk.pattern error",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(7,"tlk.sign.lost",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(8,"error from ROS",1);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(9,"if ROS in events",1);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"ROS 1",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"ROS 2",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"ROS 3",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(4,"ROS 4",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(5,"ROS 5",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(6,"ROS 6",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(7,"ROS 7",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(8,"ROS 8",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(9,"ROS 9",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(10,"ROS 10",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(11,"ROS 11",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(12,"ROS 12",2);

    histoType = "FIFOStatus";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = dbe->book2D(histoName,histoName,7,0,7,3,0,3);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"Input ch1-4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"Input ch5-8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"Input ch9-12",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(4,"Error/L1A ch1-4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(5,"Error/L1A ch5-8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(6,"Error/L1A ch9-12",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(7,"Output",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"Full",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"Almost Full",2);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"Not Full",2);	

    histoType = "L1A_IDErrorROS";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoName,12,0,12);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"ROS 1",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"ROS 2",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"ROS 3",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(4,"ROS 4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(5,"ROS 5",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(6,"ROS 6",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(7,"ROS 7",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(8,"ROS 8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(9,"ROS 9",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(10,"ROS 10",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(11,"ROS 11",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(12,"ROS 12",1);  

    histoType = "BX_IDErrorROS";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoName,12,0,12);
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(1,"ROS 1",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(2,"ROS 2",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(3,"ROS 3",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(4,"ROS 4",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(5,"ROS 5",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(6,"ROS 6",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(7,"ROS 7",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(8,"ROS 8",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(9,"ROS 9",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(10,"ROS 10",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(11,"ROS 11",1);	
    ((dduHistos[histoType])[code.getDDUID()])->setBinLabel(12,"ROS 12",1);  
  }

  // ROS Histograms
  if ( folder == "ROS_S" ) { // The summary of the error of the ROS on the same FED
    dbe->setCurrentFolder(topFolder());

    histoType = "ROSSummary";
    histoName = "FED" + dduID_s.str() + "_ROSSummary";
    string histoTitle = "Summary Wheel" + wheel_s.str() + " (FED " + dduID_s.str() + ")";

    ((rosSHistos[histoType])[code.getDDUID()]) = dbe->book2D(histoName,histoTitle,14,0,14,12,1,13);
    MonitorElement *histo = ((rosSHistos[histoType])[code.getDDUID()]);
    histo ->setBinLabel(1,"Link TimeOut",1);
    histo ->setBinLabel(2,"Ev.Id.Mis.",1);
    histo ->setBinLabel(3,"FIFO almost full",1);
    histo ->setBinLabel(4,"FIFO full",1);
    histo ->setBinLabel(5,"Ceros timeout",1);
    histo ->setBinLabel(6,"Max. wds",1);
    histo ->setBinLabel(7,"L1A FF",1);
    histo ->setBinLabel(8,"PC from TDC",1);
    histo ->setBinLabel(9,"BX ID Mis.",1);
    histo ->setBinLabel(10,"TXP",1);
    histo ->setBinLabel(11,"TDC Fatal",1);
    histo ->setBinLabel(12,"TDC FIFO Ov.",1);
    histo ->setBinLabel(13,"L1 buf. Ov.",1);
    histo ->setBinLabel(14,"L1 buf. almost full",1);
//     histo ->setBinLabel(14,"ECHO",1);

    histo ->setBinLabel(1,"ROS1",2);
    histo ->setBinLabel(2,"ROS2",2);
    histo ->setBinLabel(3,"ROS3",2);
    histo ->setBinLabel(4,"ROS4",2);
    histo ->setBinLabel(5,"ROS5",2);
    histo ->setBinLabel(6,"ROS6",2);
    histo ->setBinLabel(7,"ROS7",2);
    histo ->setBinLabel(8,"ROS8",2);
    histo ->setBinLabel(9,"ROS9",2);
    histo ->setBinLabel(10,"ROS10",2);
    histo ->setBinLabel(11,"ROS11",2);
    histo ->setBinLabel(12,"ROS12",2);
  }

  if ( folder == "ROS" ) {
    dbe->setCurrentFolder(topFolder() + "FED" + dduID_s.str() + "/" + folder + rosID_s.str());

    histoType = "ROSEventLenght";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSEventLenght";
    string histoTitle = "Event Lenght (Bytes) FED " +  dduID_s.str() + " ROS " + rosID_s.str();
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoTitle,100,0,1000);

    histoType = "ROSError";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSError";
    histoTitle = histoName + " (ROBID error summary)";
    (rosHistos[histoType])[code.getROSID()] = dbe->book2D(histoName,histoTitle,13,0,13,26,0,26);
    MonitorElement* histo = (rosHistos[histoType])[code.getROSID()];
    histo->setBinLabel(1,"Link TimeOut",1);
    histo->setBinLabel(2,"Ev.Id.Mis.",1);
    histo->setBinLabel(3,"FIFO almost full",1);
    histo->setBinLabel(4,"FIFO full",1);
    histo->setBinLabel(5,"Ceros TimeOut",1);
    histo->setBinLabel(6,"Max. wds",1);
    histo->setBinLabel(7,"L1A FF",1);
    histo->setBinLabel(8,"PC from TDC",1);
    histo->setBinLabel(9,"BX ID Mis.",1);
    histo->setBinLabel(10,"TXP",1);
    histo->setBinLabel(11,"TDC Fatal",1);
    histo->setBinLabel(12,"TDC FIFO Ov.",1);
    histo->setBinLabel(13,"L1 Buffer Ov.",1);

    histo->setBinLabel(1,"ROB0",2);
    histo->setBinLabel(2,"ROB1",2);
    histo->setBinLabel(3,"ROB2",2);
    histo->setBinLabel(4,"ROB3",2);
    histo->setBinLabel(5,"ROB4",2);
    histo->setBinLabel(6,"ROB5",2);
    histo->setBinLabel(7,"ROB6",2);
    histo->setBinLabel(8,"ROB7",2);
    histo->setBinLabel(9,"ROB8",2);
    histo->setBinLabel(10,"ROB9",2);
    histo->setBinLabel(11,"ROB10",2);
    histo->setBinLabel(12,"ROB11",2);
    histo->setBinLabel(13,"ROB12",2);
    histo->setBinLabel(14,"ROB13",2);
    histo->setBinLabel(15,"ROB14",2);
    histo->setBinLabel(16,"ROB15",2);
    histo->setBinLabel(17,"ROB16",2);
    histo->setBinLabel(18,"ROB17",2);
    histo->setBinLabel(19,"ROB18",2);
    histo->setBinLabel(20,"ROB19",2);
    histo->setBinLabel(21,"ROB20",2);
    histo->setBinLabel(22,"ROB21",2);
    histo->setBinLabel(23,"ROB22",2);
    histo->setBinLabel(24,"ROB23",2);
    histo->setBinLabel(25,"ROB24",2);
    histo->setBinLabel(26,"SC",2);

    histoType = "ROB_mean";
    histoName = "FED" + dduID_s.str() + "_" + "ROS" + rosID_s.str() + "_ROB_mean";
    string fullName = topFolder() + "FED" + dduID_s.str() + "/" + folder + rosID_s.str()+ "/" + histoName;    
    names.insert (pair<std::string,std::string> (histoType,string(fullName)));   
    (rosHistos[histoType])[code.getROSID()] = dbe->book2D(histoName,histoName,25,0,25,100,0,100);
    (rosHistos[histoType])[code.getROSID()]->setAxisTitle("ROB #",1);
    (rosHistos[histoType])[code.getROSID()]->setAxisTitle("ROB wordcounts",2);

    
    histoType = "Bunch_ID";
    histoName = "FED" + dduID_s.str() + "_" + "ROS" + rosID_s.str() + "_Bunch_ID";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,4096,0,4095);

    histoType = "Trigger_frequency";
    histoName =  "FED" + dduID_s.str() + "_Trigger_frequency"; 
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,100,1,100);
  }


  if ( folder == "TDCError") {

    dbe->setCurrentFolder(topFolder() + "FED" + dduID_s.str()+"/ROS"+rosID_s.str()+"/ROB"+robID_s.str());

    histoType = "TDCError";
    histoName = "FED" + dduID_s.str() + "_ROS" + rosID_s.str() + "_ROB"+robID_s.str()+"_TDCError";
    string histoTitle = histoName + " (TDC Errors)";
    (robHistos[histoType])[code.getROBID()] = dbe->book2D(histoName,histoTitle,3,0,3,4,0,4);

    ((robHistos[histoType])[code.getROBID()]) ->setBinLabel(1,"TDC Fatal",1);
    ((robHistos[histoType])[code.getROBID()]) ->setBinLabel(2,"TDC FIFO Ov.",1);
    ((robHistos[histoType])[code.getROBID()]) ->setBinLabel(3,"L1 Buffer Ov.",1);
    ((robHistos[histoType])[code.getROBID()]) ->setBinLabel(1,"TDC0",2);
    ((robHistos[histoType])[code.getROBID()]) ->setBinLabel(2,"TDC1",2);
    ((robHistos[histoType])[code.getROBID()]) ->setBinLabel(3,"TDC2",2);
    ((robHistos[histoType])[code.getROBID()]) ->setBinLabel(4,"TDC3",2);

  }

  // SC Histograms
  if ( folder == "SC" ) {
    // Same numbering for SC as for ROS
    dbe->setCurrentFolder(topFolder() + "FED" + dduID_s.str() + "/" + folder + rosID_s.str());

    // the SC histos belong to the ROS map (pay attention) since the data come from the corresponding ROS

    histoType = "SCTriggerBX";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_SCTriggerBX";
    string histoTitle = histoName + " (station vs BX)";
    (rosHistos[histoType])[code.getSCID()] = dbe->book2D(histoName,histoTitle,128,0,128,4,1,5);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(1,"MB1",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(2,"MB2",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(3,"MB3",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(4,"MB4",2);


    histoType = "SCTriggerQuality";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_SCTriggerQuality";
    histoTitle = histoName + "(quality vs station)";
    (rosHistos[histoType])[code.getSCID()] = dbe->book2D(histoName,histoTitle,4,1,5,8,0,8);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(1,"MB1",1);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(2,"MB2",1);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(3,"MB3",1);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(4,"MB4",1);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(1,"Li",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(2,"Lo",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(3,"Hi",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(4,"Ho",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(5,"LL",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(6,"HL",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(7,"HH",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(8,"Null",2);
    
  }
}

void DTDataIntegrityTask::TimeHistos(string histoType){  
  
 if(histoType == "Event_word_vs_time"){   

  for (it = names.begin(); it != names.end(); it++) {    

    if ((*it).first==histoType){
     
     MonitorElement * h1 =dbe->get((*it).second);

 int first_bin = -1, last_bin=-1;
   for( int bin=1; bin < h1->getNbinsX()+1; bin++ ){
    for( int j=1; j < h1->getNbinsY(); j++ ){
     if( h1->getBinContent(bin,j) > 0 ) {    
      if( first_bin == -1 ) { first_bin = bin; }
      last_bin = bin;
   }
  }
 }
 
  if( first_bin > 1 ) { first_bin -= 1; }
  if( last_bin < h1-> getNbinsX() ){ last_bin += 1; }
    h1->setAxisRange(0,last_bin,1);
   }
  }
 }  
}



// void DTDataIntegrityTask::bookHistosFED() {
//     bookHistos( string("ROS_S"), code);

// }


void DTDataIntegrityTask::bookHistosROS25(DTROChainCoding code) {
    bookHistos( string("ROS"), code);
    for(int robId = 0; robId != 25; ++robId) {
      code.setROB(robId);
      bookHistos( string("TDCError"), code);
    }
    bookHistos( string("SC"), code);
}


void DTDataIntegrityTask::processROS25(DTROS25Data & data, int ddu, int ros) {
  neventsROS25++; // FIXME: implement a counter which makes sense

  if (neventsROS25%1000 == 0)
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< "[DTDataIntegrityTask]: " << neventsROS25 << " events analyzed by processROS25" << endl;

  // The ID of the RO board (used to map the histos)
  DTROChainCoding code;
  code.setDDU(ddu);
  code.setROS(ros);

  MonitorElement* ROSSummary = rosSHistos["ROSSummary"][code.getDDUID()];

  // Summary of all ROB errors
  MonitorElement* ROSError = 0;
  if(!hltMode) ROSError = rosHistos["ROSError"][code.getROSID()];



  // ROS errors


  // check for TPX errors
  if (data.getROSTrailer().TPX() != 0) {
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << " TXP error en ROS "
								      << code.getROS() << endl;
    ROSSummary->Fill(9,code.getROS());
  }

  // L1 Buffer almost full (non-critical error!)
  // equivalent to (data.getROSTrailer().ECHO() >>1) & 0x1 == 1
  if(data.getROSTrailer().ECHO() > 1) {
    ROSSummary->Fill(13,code.getROS());
  }
  
  // FIXME: what is this about???
  if (neventsROS25 == 1) FirstRos = code.getROSID();
  if (code.getROSID() == FirstRos) nevents++ ;


  for (vector<DTROSErrorWord>::const_iterator error_it = data.getROSErrors().begin();
       error_it != data.getROSErrors().end(); error_it++) { // Loop over ROS error words

    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << " Error in ROS " << code.getROS()
      << " ROB Id " << (*error_it).robID()
      << " Error type " << (*error_it).errorType() << endl;

    // Fill the ROSSummary (1 per FED) histo
    ROSSummary->Fill((*error_it).errorType(), code.getROS());
    if((*error_it).errorType() <= 11) { // set error flag
       eventErrorFlag = true;
    }
    
    if(!hltMode) {
      // Fill the ROB Summary (1 per ROS) histo
      if ((*error_it).errorType() != 4) {
	ROSError->Fill((*error_it).errorType(),(*error_it).robID());
      }
    }
  }


  int ROSDebug_BunchNumber = -1;
  int ROSDebug_BcntResCntLow = 0;
  int ROSDebug_BcntResCntHigh = 0;
  int ROSDebug_BcntResCnt = 0;
  
  for (vector<DTROSDebugWord>::const_iterator debug_it = data.getROSDebugs().begin();
       debug_it != data.getROSDebugs().end(); debug_it++) { // Loop over ROS debug words
    
    if ((*debug_it).debugType() == 0 ) {
      ROSDebug_BunchNumber = (*debug_it).debugMessage();
    } else if ((*debug_it).debugType() == 1 ) {
      ROSDebug_BcntResCntLow = (*debug_it).debugMessage();
    } else if ((*debug_it).debugType() == 2 ) {
      ROSDebug_BcntResCntHigh = (*debug_it).debugMessage();
    }
  }

  ROSDebug_BcntResCnt = (ROSDebug_BcntResCntHigh << 15) + ROSDebug_BcntResCntLow;
  //   LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
  //     << " ROS: " << code.getROS() << " ROSDebug_BunchNumber " << ROSDebug_BunchNumber
  //     << " ROSDebug_BcntResCnt " << ROSDebug_BcntResCnt << endl;
  

  //	 Event words vs time
  // FIXME: what is this doing???
  ROSWords_t(ResetCount_unfolded,code.getROS(),ROSDebug_BcntResCnt,nevents);

  // fill hists it here
  //   histoType = "Event_word_vs_time";  	  
  //   if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()){
  //   (rosHistos.find(histoType)->second).find(code.getROSID())->second->
  //   		Fill((ResetCount_unfolded),data.getROSTrailer().EventWordCount());
  //   (rosHistos.find(histoType)->second).find(code.getROSID())->second->setAxisTitle("Time(s)",1);
  //    }
  //   else {
  //      (rosHistos.find(histoType)->second).find(code.getROSID())->second->
  //     		Fill((ResetCount_unfolded),data.getROSTrailer().EventWordCount());}  


	

  // ROB Group Header
  for (vector<DTROBHeader>::const_iterator rob_it = data.getROBHeaders().begin();
       rob_it != data.getROBHeaders().end(); rob_it++) { // loop over ROB headers
    
    code.setROB((*rob_it).first);
    DTROBHeaderWord robheader = (*rob_it).second;  

    if(!hltMode) rosHistos["Bunch_ID"][code.getROSID()]->Fill(robheader.bunchID());
    
    if (robheader.bunchID() != ROSDebug_BunchNumber) {
      // fill ROS Summary plot
      ROSSummary->Fill(8,code.getROS());
      eventErrorFlag = true;
      
      // fill ROB Summary plot for that particular ROS
      if(!hltMode) ROSError->Fill(8,robheader.robID());
    }
  }


  if(!hltMode) { // produce only when not in HLT 
    // ROB Trailer
    for (vector<DTROBTrailerWord>::const_iterator robt_it = data.getROBTrailers().begin();
	 robt_it != data.getROBTrailers().end(); robt_it++) { // loop over ROB trailers 
    
      rosHistos["ROB_mean"][code.getROSID()]->Fill(code.getROB(),(*robt_it).wordCount());
    }

    // Trigger frequency
    double frequency = 0;
    // FIXME: how is the frequency computed
    ROS_L1A_Frequency(code.getROS(),ROSDebug_BcntResCnt,neventsROS25,frequency,trigger_counter);
    rosHistos["Trigger_frequency"][code.getROSID()]->Fill(frequency);

    // Plot the event lenght //NOHLT
    rosHistos["ROSEventLenght"][code.getROSID()]->Fill(data.getROSTrailer().EventWordCount()*4);
  }

   
  // TDC Data  
  for (vector<DTTDCData>::const_iterator tdc_it = data.getTDCData().begin();
       tdc_it != data.getTDCData().end(); tdc_it++) { // loop over TDC data

    DTTDCMeasurementWord tdcDatum = (*tdc_it).second;

    if ( tdcDatum.PC() !=0)  {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " PC error in ROS " << code.getROS() << " TDC " << (*tdc_it).first << endl;
      //     fill ROS Summary plot
      ROSSummary->Fill(7,code.getROS());

      eventErrorFlag = true;

      // fill ROB Summary plot for that particular ROS
      if(!hltMode) ROSError->Fill(7,(*tdc_it).first);
    }
  }

  // TDC Error  
  for (vector<DTTDCError>::const_iterator tdc_it = data.getTDCError().begin();
       tdc_it != data.getTDCError().end(); tdc_it++) { // loop over TDC errors

    code.setROB((*tdc_it).first);

    float type_TDC_error_for_plot_1 = 0;
    float type_TDC_error_for_plot_2 = 0;

    if(((*tdc_it).second).tdcError() & 0x4000 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " ROS " << code.getROS() << " ROB " << code.getROB()
	<< " Internal fatal Error 4000 in TDC " << (*tdc_it).first << endl;

      type_TDC_error_for_plot_1 = 10;
      type_TDC_error_for_plot_2 = 0;

    } else if ( ((*tdc_it).second).tdcError() & 0x1b6d ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " ROS " << code.getROS() << " ROB " << code.getROB()
	<< " TDC FIFO full in TDC " << (*tdc_it).first << endl;

      type_TDC_error_for_plot_1 = 11;
      type_TDC_error_for_plot_2 = 1;

    } else if ( ((*tdc_it).second).tdcError() & 0x2492 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " ROS " << code.getROS() << " ROB " << code.getROB()
	<< " L1 buffer overflow in TDC " << (*tdc_it).first << endl;
      
      type_TDC_error_for_plot_1 = 12;
      type_TDC_error_for_plot_2 = 2;

    } else {
      LogWarning("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " TDC error code not known " << ((*tdc_it).second).tdcError() << endl;
    }
    
    ROSSummary->Fill(type_TDC_error_for_plot_1,code.getROS());

    if(type_TDC_error_for_plot_1 <= 11) {
      eventErrorFlag = true;
    }

    if(!hltMode) {
      ROSError->Fill(type_TDC_error_for_plot_1,(*tdc_it).first);
      robHistos["TDCError"][code.getROBID()]->Fill(type_TDC_error_for_plot_2,((*tdc_it).second).tdcID());
    }
  }

  // Read SC data
  if (!hltMode && getSCInfo) {
    // SC Data
    int stationGroup = 0 ; //= ((*sc_it).second)%2;
    for (vector<DTSectorCollectorData>::const_iterator sc_it = data.getSCData().begin();
	 sc_it != data.getSCData().end(); sc_it++) { // loop over SC data

      // SC Data words are devided into 2 parts each of 8 bits:
      //  LSB refers to MB1 and MB3
      //  MSB refers to MB2 and MB4

      // fill only the information regarding SC words with trigger
      bool hasTrigger_LSB = ((*sc_it).first).hasTrigger(0);
      bool hasTrigger_MSB = ((*sc_it).first).hasTrigger(1);

      // the quality
      int quality_LSB = ((*sc_it).first).trackQuality(0);
      int quality_MSB = ((*sc_it).first).trackQuality(1);

      if (hasTrigger_LSB) {

	rosHistos["SCTriggerBX"][code.getSCID()]->Fill((*sc_it).second, 1+stationGroup*2);
	rosHistos["SCTriggerQuality"][code.getSCID()]->Fill(1+stationGroup*2,quality_LSB);

      }

      if (hasTrigger_MSB) {
	rosHistos["SCTriggerBX"][code.getSCID()]->Fill((*sc_it).second, 2+stationGroup*2);
	rosHistos["SCTriggerQuality"][code.getSCID()]->Fill(2+stationGroup*2,quality_MSB);

      }
      stationGroup = (stationGroup == 0 ? 1 : 0);  //switch between MB1-2 and MB3-4 data
    }
  }
}

void DTDataIntegrityTask::processFED(DTDDUData & data, const std::vector<DTROS25Data> & rosData, int ddu) {

  neventsDDU++;
  if (neventsDDU%1000 == 0)
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << "[DTDataIntegrityTask]: " << neventsDDU << " events analyzed by processFED" << endl;

  if(hltMode) return;

  DTROChainCoding code;
  code.setDDU(ddu);


  FEDTrailer trailer = data.getDDUTrailer();
  FEDHeader header = data.getDDUHeader();
  // FIXME: add a check on the header and trailer?
  //   if(!trailer.check()) -> log in an histo
  
  DTDDUSecondStatusWord secondWord = data.getSecondStatusWord();


  //1D HISTO WITH TTS VALUES form trailer (7 bins = 7 values)
  MonitorElement* hTTSValue = dduHistos["TTSValues"][code.getDDUID()];

  
  int ttsCodeValue = -1;
  switch(trailer.ttsBits()){
  case 0:{ //disconnected
    ttsCodeValue = 0;
    break;
  }
  case 1:{ //warning overflow
    ttsCodeValue = 1;
    break;
  }
  case 2:{ //out of sinch
    ttsCodeValue = 2;
    break;
  }
  case 4:{ //busy
    ttsCodeValue = 3;
    break;
  }
  case 8:{ //ready
    ttsCodeValue = 4;
    break;
  }
  case 12:{ //error
    ttsCodeValue = 5;
    break;
  }
  case 16:{ //disconnected
    ttsCodeValue = 6;
    break;
  }
  default:{
    LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      <<"[DTDataIntegrityTask] DDU control: wrong TTS value "<<trailer.ttsBits()<<endl;
    // FIXME: add a bin to the histo?
  }
  }
  hTTSValue->Fill(ttsCodeValue);



  //1D HISTO: IF TTS=2,12 CHECK L1A AND BX MISIMATCH, FIFO AND ROS ERROR (from status words)
  string histoType;
  if(trailer.ttsBits()==2){
    histoType = "TTS_2";
  }
  if(trailer.ttsBits()==12){
    histoType = "TTS_12";
  }

  if(trailer.ttsBits()==2 || trailer.ttsBits()==12) {
    MonitorElement *hTTS_2or12 = dduHistos[histoType][code.getDDUID()];
    hTTS_2or12->Fill(0,secondWord.l1AIDError());
    hTTS_2or12->Fill(1,secondWord.bxIDError());
    hTTS_2or12->Fill(2,(secondWord.fifoFull() & 0x1));
    hTTS_2or12->Fill(3,(secondWord.fifoFull() & 0x2));
    hTTS_2or12->Fill(4,(secondWord.fifoFull() & 0x4));
    hTTS_2or12->Fill(5,(secondWord.inputFifoFull() & 0x1));
    hTTS_2or12->Fill(6,(secondWord.inputFifoFull() & 0x2));
    hTTS_2or12->Fill(7,(secondWord.inputFifoFull() & 0x4));
    hTTS_2or12->Fill(8,secondWord.outputFifoFull());
    int channel=1;
    for (vector<DTDDUFirstStatusWord>::const_iterator fsw_it = data.getFirstStatusWord().begin();
	 fsw_it != data.getFirstStatusWord().end(); fsw_it++) {
      if((*fsw_it).timeout() || (*fsw_it).eventTrailerLost() || (*fsw_it).opticalFiberSignalLost() ||
	 (*fsw_it).tlkPropagationError()||(*fsw_it).tlkPatternError() ||(*fsw_it).tlkSignalLost() || (*fsw_it).errorFromROS())
	hTTS_2or12->Fill(8+channel,1);
      channel++;
    }
  }

  //MONITOR TTS VS TIME 
  //   pair<int,int> ev_tts= make_pair(header.lvl1ID(),trailer.ttsBits());
  //   //insert the pair at the right position
  //   for (list<pair<int,int> >::iterator ev_it = ttsVSTime.begin(); ; ev_it++) {
  //     if(ev_it == ttsVSTime.end()){
  //       ttsVSTime.push_back(ev_tts);
  //       break;
  //     }
  //     else if(header.lvl1ID() < (*ev_it).first) {
  //       ttsVSTime.insert(ev_it, ev_tts);
  //       break;
  //     }
  //   }
  //   //loop until the event number are sequential
  //   if(!(header.lvl1ID() % 10)){
  //     //create a copy of the list to remove elements already analyzed
  //     list<pair<int,int> > ttsVSTime_copy(ttsVSTime);
  //     int counter_ev=myPrevEv;
  //       for (list<pair<int,int> >::iterator ev_it = ttsVSTime.begin(); ; ev_it++) {
  // 	counter_ev++;

  // 	if((*ev_it).first != counter_ev || ev_it == ttsVSTime.end())
  // 	  break;

  // 	if((*ev_it).first > myPrevEv){
  // 	  myPrevEv = (*ev_it).first;

  // 	  //add a point if the value is changed
  // 	  if((*ev_it).second != myPrevTtsVal){
  // 	    //graphTTS->addPoint
  // 	    myPrevTtsVal = (*ev_it).second;
  // 	  }
  // 	}

  // 	//remove from the list the ordered events already analyzed
  // 	list<pair<int,int> >::iterator copy_it = ev_it;
  // 	ttsVSTime_copy.remove(*copy_it);
  //       }
  //       ttsVSTime.clear();
  //       ttsVSTime.merge(ttsVSTime_copy);
  //   }

  //1D HISTOS: EVENT LENGHT from trailer
  //cout<<"1D HISTOS WITH EVENT LENGHT from trailer"<<endl;
  dduHistos["EventLenght"][code.getDDUID()]->Fill(trailer.lenght()*8);

  //1D HISTO: EVENT TYPE from header
  //cout<<"1D HISTO WITH EVENT TYPE from header"<<endl;
  dduHistos["EventType"][code.getDDUID()]->Fill(header.triggerType());

  //1D HISTO: NUMBER OF ROS IN THE EVENTS from 2nd status word
  int rosList = secondWord.rosList();
  vector<int> rosPositions;
  for(int i=0;i<12;i++){
    if(rosList & 0x1)
      rosPositions.push_back(i);
    rosList >>= 1;
  }

  dduHistos["ROSList"][code.getDDUID()]->Fill(rosPositions.size());

  //2D HISTO: ROS VS STATUS (8 BIT = 8 BIN) from 1st-2nd status words (9th BIN FROM LIST OF ROS in 2nd status word)
  MonitorElement* hROSStatus = dduHistos["ROSStatus"][code.getDDUID()];
  int channel=0;
  for (vector<DTDDUFirstStatusWord>::const_iterator fsw_it = data.getFirstStatusWord().begin();
       fsw_it != data.getFirstStatusWord().end(); fsw_it++) {
    // assuming association one-to-one between DDU channel and ROS
    hROSStatus->Fill(0,channel,(*fsw_it).channelEnabled());
    hROSStatus->Fill(1,channel,(*fsw_it).timeout());
    hROSStatus->Fill(2,channel,(*fsw_it).eventTrailerLost());
    hROSStatus->Fill(3,channel,(*fsw_it).opticalFiberSignalLost());
    hROSStatus->Fill(4,channel,(*fsw_it).tlkPropagationError());
    hROSStatus->Fill(5,channel,(*fsw_it).tlkPatternError());
    hROSStatus->Fill(6,channel,(*fsw_it).tlkSignalLost());
    hROSStatus->Fill(7,channel,(*fsw_it).errorFromROS());
    channel++;
  }
  //9th BIN FROM LIST OF ROS in 2nd status word
  for(vector<int>::const_iterator channel_it = rosPositions.begin(); channel_it != rosPositions.end(); channel_it++){
    hROSStatus->Fill(8,(*channel_it),1);
  }

  //MONITOR ROS LIST VS TIME 
  //  pair<int,int> ev_ros= make_pair(header.lvl1ID(),rosPositions.size());
  //   //insert the pair at the right position
  //   for (list<pair<int,int> >::iterator ev_it = rosVSTime.begin(); ; ev_it++) {
  //     if(ev_it == rosVSTime.end()){
  //       rosVSTime.push_back(ev_ros);
  //       break;
  //     }
  //     else if(header.lvl1ID() < (*ev_it).first) {
  //       rosVSTime.insert(ev_it, ev_ros);
  //       break;
  //     }
  //   }

  //   //loop until the last sequential event number (= myPrevEv set by loop on ttsVSTime)
  //   if(!(header.lvl1ID() % 10)){
  //     //create a copy of the list to remove elements already analyzed
  //     list<pair<int,int> > rosVSTime_copy(rosVSTime);
  //     for (list<pair<int,int> >::iterator ev_it = rosVSTime.begin(); ; ev_it++) {
      
  //       if((*ev_it).first > myPrevEv || ev_it == rosVSTime.end())
  // 	break;
      
  //       //add a point if the value is changed
  //       if((*ev_it).second != myPrevRosVal){
  // 	//graphROS->addPoint
  // 	myPrevRosVal = (*ev_it).second;
  //      }
  //       //remove from the list the ordered events already analyzed
  //       list<pair<int,int> >::iterator copy_it = ev_it;
  //       rosVSTime_copy.remove(*copy_it);
  //     }
  //     rosVSTime.clear();
  //     rosVSTime.merge(rosVSTime_copy);
  //   }

  //2D HISTO: FIFO STATUS from 2nd status word
  MonitorElement *hFIFOStatus = dduHistos["FIFOStatus"][code.getDDUID()];
  int fifoStatus[7]; //Input*3,L1A*3,Output with value 0=full,1=AlmostFull,2=NotFull
  int inputFifoFull = secondWord.inputFifoFull();
  int inputFifoAlmostFull = secondWord.inputFifoAlmostFull();
  int fifoFull = secondWord.fifoFull();
  int fifoAlmostFull = secondWord.fifoAlmostFull();
  int outputFifoFull = secondWord.outputFifoFull();
  int outputFifoAlmostFull = secondWord.outputFifoAlmostFull();
  for(int i=0;i<3;i++){
    if(inputFifoFull & 0x1){
      fifoStatus[i]=0;
      hFIFOStatus->Fill(i,0);
    }
    if(inputFifoAlmostFull & 0x1){
      fifoStatus[i]=1;
      hFIFOStatus->Fill(i,1);
    }
    if(fifoFull & 0x1){
      fifoStatus[3+i]=0;
      hFIFOStatus->Fill(3+i,0);
    }
    if(fifoAlmostFull & 0x1){
      fifoStatus[3+i]=1;
      hFIFOStatus->Fill(3+i,1);
    }
    if(!(inputFifoFull & 0x1) && !(inputFifoAlmostFull & 0x1)){
      fifoStatus[i]=2;
      hFIFOStatus->Fill(i,2);
    }
    if(!(fifoFull & 0x1) && !(fifoAlmostFull & 0x1)){
      fifoStatus[3+i]=2;
      hFIFOStatus->Fill(3+i,2);
    }
    inputFifoFull >>= 1;
    inputFifoAlmostFull >>= 1;
    fifoFull >>= 1;
    fifoAlmostFull >>= 1;
  }

  if(outputFifoFull){
    fifoStatus[6]=0;
    hFIFOStatus->Fill(6,0);
  }
  if(outputFifoAlmostFull){
    fifoStatus[6]=1;
    hFIFOStatus->Fill(6,1);
  }
  if(!outputFifoFull && !outputFifoAlmostFull){
    fifoStatus[6]=2;
    hFIFOStatus->Fill(6,2);
  }

  //MONITOR FIFO VS TIME 
  // pair<int,int*> ev_fifo= make_pair(header.lvl1ID(),fifoStatus);
  //   //insert the pair at the right position
  //   for (list<pair<int,int*> >::iterator ev_it = fifoVSTime.begin(); ; ev_it++) {
  //     if(ev_it == fifoVSTime.end()){
  //       fifoVSTime.push_back(ev_fifo);
  //       break;
  //     }
  //     else if(header.lvl1ID() < (*ev_it).first) {
  //       fifoVSTime.insert(ev_it, ev_fifo);
  //       break;
  //     }
  //   }

  //   //loop until the last sequential event number (= myPrevEv set by loop on ttsVSTime)
  //   if(!(header.lvl1ID() % 10)){
  //     //create a copy of the list to remove elements already analyzed
  //     list<pair<int,int*> > fifoVSTime_copy(fifoVSTime);
  //     for (list<pair<int,int*> >::iterator ev_it = fifoVSTime.begin(); ; ev_it++) {
  //       if((*ev_it).first > myPrevEv || ev_it == fifoVSTime.end())
  // 	break;
      
  //       //add a point if one of the values is changed
  //       for(int i=0; i<7; i++){
  // 	if((*ev_it).second[i] != myPrevFifoVal[i]){
  // 	  //graphFIFO[i]->addPoint
  // 	  myPrevFifoVal[i] = (*ev_it).second[i];
  // 	}
  //       }
  //       //remove from the list the ordered events already analyzed
  //       list<pair<int,int*> >::iterator copy_it = ev_it;
  //       fifoVSTime_copy.remove(*copy_it);
  //     }
  //     fifoVSTime.clear();
  //     fifoVSTime.merge(fifoVSTime_copy);
  //   }


  if(trailer.ttsBits()==2) {   //DDU OUT OF SYNCH

    //If BX_ID error identify which ROS has wrong BX
    MonitorElement *hBX_IDErrorROS = dduHistos["BX_IDErrorROS"][code.getDDUID()];
    for (vector<DTROS25Data>::const_iterator ros_it = rosData.begin();
	 ros_it != rosData.end(); ros_it++) {
      for (vector<DTROSDebugWord>::const_iterator debug_it = (*ros_it).getROSDebugs().begin();
	   debug_it != (*ros_it).getROSDebugs().end(); debug_it++) {
	if ((*debug_it).debugType() == 0 ) {
	  int ROSDebug_BXID = (*debug_it).debugMessage();
	  if(ROSDebug_BXID != header.bxID()) {
	    hBX_IDErrorROS->Fill((*ros_it).getROSID()-1);
	    LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	      << "BX_ID error from ROS "<<(*ros_it).getROSID()<<" :"
	      <<" ROSDebug_BXID "<< ROSDebug_BXID
	      <<"   DDUHeader_BXID "<< header.bxID()<<endl;
	  }
	}
      }
    }

    //If L1A_ID error identify which ROS has wrong L1A 
    for (vector<DTROS25Data>::const_iterator ros_it = rosData.begin();
	 ros_it != rosData.end(); ros_it++) {
      int ROSHeader_TTCCount = ((*ros_it).getROSHeader()).TTCEventCounter();
      if(ROSHeader_TTCCount != header.lvl1ID()-1){
	dduHistos["L1A_IDErrorROS"][code.getDDUID()]->Fill((*ros_it).getROSID()-1);
	LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	  << "L1A_ID error from ROS "<<(*ros_it).getROSID()<<" :"
	  <<" ROSHeader_TTCeventcounter " << ROSHeader_TTCCount
	  <<"   DDUHeader_lvl1ID "<< header.lvl1ID()<<endl;
      }
    }
  }
}

  
bool DTDataIntegrityTask::eventHasErrors() const {
  return eventErrorFlag;
}



// log number of times the payload of each fed is unpacked 
void DTDataIntegrityTask::fedEntry(int dduID) {
  hFEDEntry->Fill(dduID);
}



// log number of times the payload of each fed is skipped (no ROS inside)
void DTDataIntegrityTask::fedFatal(int dduID) {
  hFEDFatal->Fill(dduID);
}



// log number of times the payload of each fed is partially skipped (some ROS skipped)
void DTDataIntegrityTask::fedNonFatal(int dduID) {
  hFEDNonFatal->Fill(dduID);
}



std::string DTDataIntegrityTask::topFolder() const {
  if(hltMode) return string("DT/00-DataIntegrity_EvF/");
  return string("DT/00-DataIntegrity/");
}



void DTDataIntegrityTask::preProcessEvent(const edm::EventID& iEvtid, const edm::Timestamp& iTime) {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: preProcessEvent" <<endl;

  // reset the error flag
  eventErrorFlag = false;
}



void DTDataIntegrityTask::postBeginJob() {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: postBeginJob" <<endl;
  // get the DQMStore service if needed
  dbe = edm::Service<DQMStore>().operator->();    
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask] Get DQMStore service" << endl;
  
  // book FED integrity histos
  bookHistos();

  
  // Loop over the DT FEDs
  int FEDIDmin = FEDNumbering::getDTFEDIds().first;
  int FEDIDMax = FEDNumbering::getDTFEDIds().second;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << " FEDS: " << FEDIDmin  << " to " <<  FEDIDMax << " in the RO" << endl;


  // static booking of the histograms
  for(int fed = FEDIDmin; fed < FEDIDMax; ++fed) { // loop over the FEDs in the readout
    DTROChainCoding code;
    code.setDDU(fed);
    
    bookHistos( string("ROS_S"), code);

    // if in HLT book only the summaries and the FEDIntegrity histos
    if(hltMode) continue;

    bookHistos( string("DDU"), code);

    for(int ros = 1; ros <= 12; ++ros) {// loop over all ROS
      code.setROS(ros);
      bookHistosROS25(code);
    }
  }

}
