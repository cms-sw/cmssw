
/*
 * \file DTDataIntegrityTask.cc
 * 
 * $Date: 2008/04/23 13:33:14 $
 * $Revision: 1.41 $
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

DTDataIntegrityTask::DTDataIntegrityTask(const edm::ParameterSet& ps,edm::ActivityRegistry& reg) {

  reg.watchPostEndJob(this,&DTDataIntegrityTask::postEndJob);
  
  debug = ps.getUntrackedParameter<bool>("debug", false);
  if (debug)
    cout<<"[DTDataIntegrityTask]: Constructor"<<endl;

  neventsDDU = 0;
  neventsROS25 = 0;

  parameters = ps;

  dbe = edm::Service<DQMStore>().operator->();  

  doTimeHisto =  ps.getUntrackedParameter<bool>("doTimeHisto", true);
}



DTDataIntegrityTask::~DTDataIntegrityTask() {
  if(debug)
    cout<<"[DTDataIntegrityTask]: Destructor. Analyzed "<< neventsDDU <<" events"<<endl;
}


void DTDataIntegrityTask::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context) {

  if(debug)
    cout<<"[DTDataIntegrityTask]: Begin of LS transition"<<endl;

  if(lumiSeg.id().luminosityBlock()%parameters.getUntrackedParameter<int>("ResetCycle", 3) == 0) {
    for(map<string, map<int, MonitorElement*> > ::const_iterator ddu_histo = dduHistos.begin();
	ddu_histo != dduHistos.end();
	ddu_histo++) {
      for(map<int, MonitorElement*> ::const_iterator dh = (*ddu_histo).second.begin();
	  dh != (*ddu_histo).second.end();
	  dh++) {
	(*dh).second->Reset();
      }
    }
    for(map<string, map<int, MonitorElement*> > ::const_iterator rosS_histo = rosSHistos.begin();
	rosS_histo != rosSHistos.end();
	rosS_histo++) {
       for(map<int, MonitorElement*> ::const_iterator rosS = (*rosS_histo).second.begin();
	  rosS != (*rosS_histo).second.end();
	  rosS++) {
	 (*rosS).second->Reset();
       }
    }
    for(map<string, map<int, MonitorElement*> > ::const_iterator ros_histo = rosHistos.begin();
	ros_histo != rosHistos.end();
	ros_histo++) {
      for(map<int, MonitorElement*> ::const_iterator rosh = (*ros_histo).second.begin();
	  rosh != (*ros_histo).second.end();
	  rosh++) {
	(*rosh).second->Reset();
      }
    }
    for(map<string, map<int, MonitorElement*> > ::const_iterator rob_histo = robHistos.begin();
	rob_histo != robHistos.end();
	rob_histo++) {
      for(map<int, MonitorElement*> ::const_iterator robh = (*rob_histo).second.begin();
	  robh != (*rob_histo).second.end();
	  robh++) {
	(*robh).second->Reset();
      }
    }
  }

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
//  if(debug)
    cout<<"[DTDataIntegrityTask]: postEndJob called!"<<endl;

  if(doTimeHisto) TimeHistos("Event_word_vs_time");	
	
  dbe->rmdir("DT/DataIntegrity");

}

void DTDataIntegrityTask::bookHistos(string folder, DTROChainCoding code) {

  stringstream dduID_s; dduID_s << code.getDDU();
  stringstream rosID_s; rosID_s << code.getROS();
  stringstream robID_s; robID_s << code.getROB();

  string histoType;
  string histoName;

  // DDU Histograms
  if ( folder == "DDU" ) {
    dbe->setCurrentFolder("DT/DataIntegrity/FED" + dduID_s.str());

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
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoName,1000,0,1000);
 
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
    (dduHistos[histoType])[code.getDDUID()] = dbe->book1D(histoName,histoName,12,0,12);
    
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

  if ( folder == "ROS_S" ) {
    dbe->setCurrentFolder("DT/DataIntegrity/FED" + dduID_s.str());

    histoType = "ROSSummary";
    histoName = "FED" + dduID_s.str() + "_ROSSummary";

    ((rosSHistos[histoType])[code.getDDUID()]) = dbe->book2D(histoName,histoName,13,0,13,12,1,13);

    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(1,"Link TimeOut",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(2,"Ev.Id.Mis.",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(3,"FIFO almost full",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(4,"FIFO full",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(5,"Ceros TimeOut",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(6,"Max. wds",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(7,"L1A FF",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(8,"PC from TDC",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(9,"BX ID Mis.",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(10,"TXP",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(11,"TDC Fatal",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(12,"TDC FIFO Ov.",1);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(13,"L1 Buffer Ov.",1);

    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(1,"ROS1",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(2,"ROS2",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(3,"ROS3",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(4,"ROS4",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(5,"ROS5",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(6,"ROS6",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(7,"ROS7",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(8,"ROS8",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(9,"ROS9",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(10,"ROS10",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(11,"ROS11",2);
    ((rosSHistos[histoType])[code.getDDUID()]) ->setBinLabel(12,"ROS12",2);
  }

  if ( folder == "ROS" ) {

    dbe->setCurrentFolder("DT/DataIntegrity/FED" + dduID_s.str() + "/" + folder + rosID_s.str());

    histoType = "ROSEventLenght";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSEventLenght";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,100,0,1000);

    histoType = "ROSTrailerBits";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSTrailerBits";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,128,0,128);
 

    histoType = "ROSError";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSError";
    string histoTitle = histoName + " (ROBID error summary)";
    (rosHistos[histoType])[code.getROSID()] = dbe->book2D(histoName,histoTitle,13,0,13,26,0,26);
    
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(1,"Link TimeOut",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(2,"Ev.Id.Mis.",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(3,"FIFO almost full",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(4,"FIFO full",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(5,"Ceros TimeOut",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(6,"Max. wds",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(7,"L1A FF",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(8,"PC from TDC",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(9,"BX ID Mis.",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(10,"TXP",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(11,"TDC Fatal",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(12,"TDC FIFO Ov.",1);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(13,"L1 Buffer Ov.",1);

    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(1,"ROB0",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(2,"ROB1",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(3,"ROB2",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(4,"ROB3",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(5,"ROB4",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(6,"ROB5",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(7,"ROB6",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(8,"ROB7",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(9,"ROB8",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(10,"ROB9",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(11,"ROB10",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(12,"ROB11",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(13,"ROB12",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(14,"ROB13",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(15,"ROB14",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(16,"ROB15",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(17,"ROB16",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(18,"ROB17",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(19,"ROB18",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(20,"ROB19",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(21,"ROB20",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(22,"ROB21",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(23,"ROB22",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(24,"ROB23",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(25,"ROB24",2);
    ((rosHistos[histoType])[code.getROSID()]) ->setBinLabel(26,"SC",2);

    histoType = "ROSDebug_BunchNumber";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSDebug_BunchNumber";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,3564,0,3564);

    histoType = "ROSDebug_BcntResCnt";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSDebug_BcntResCnt";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,16384,0,65536);

    histoType = "Event_word_vs_time";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_Event_word_vs_time";
    string fullName = "DT/DataIntegrity/FED" + dduID_s.str() + "/" + folder + rosID_s.str()+ "/" + histoName;
    names.insert (pair<std::string,std::string> (histoType,string(fullName)));
    (rosHistos[histoType])[code.getROSID()] = dbe->book2D(histoName,histoName,1440,0,28800,100,0,3000);    
    
    histoType = "ROB_mean";
    histoName = "FED" + dduID_s.str() + "_" + "ROS" + rosID_s.str() + "_ROB_mean";
    fullName = "DT/DataIntegrity/FED" + dduID_s.str() + "/" + folder + rosID_s.str()+ "/" + histoName;    
    names.insert (pair<std::string,std::string> (histoType,string(fullName)));   
    (rosHistos[histoType])[code.getROSID()] = dbe->book2D(histoName,histoName,25,0,25,100,0,100);
    
    histoType = "Bunch_ID";
    histoName = "FED" + dduID_s.str() + "_" + "ROS" + rosID_s.str() + "_Bunch_ID";
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,4096,0,4095);

    histoType = "Trigger_frequency";
    histoName =  "FED" + dduID_s.str() + "_Trigger_frequency"; 
    (rosHistos[histoType])[code.getROSID()] = dbe->book1D(histoName,histoName,100,1,100);
         }

  // ROB/TDC Histograms
  if ( folder == "ROB_O") {
    
    dbe->setCurrentFolder("DT/DataIntegrity/FED" + dduID_s.str()+"/ROS"+rosID_s.str()+"/ROB"+robID_s.str());

    histoType = "Occupancy";
    histoName = "FED" + dduID_s.str() + "_ROS" + rosID_s.str() + "_ROB"+robID_s.str()+"_Occupancy";
    string histoTitle = histoName + " (TDC vs TDCchannel)";
    (robHistos[histoType])[code.getROBID()] = dbe->book2D(histoName,histoTitle,32,0,32,4,0,4);
   }

  if ( folder == "ROB_T") {

    dbe->setCurrentFolder("DT/DataIntegrity/FED" + dduID_s.str()+"/ROS"+rosID_s.str()+"/ROB"+robID_s.str());

    histoType = "TimeBox";
    histoName = "FED" + dduID_s.str() + "_ROS" + rosID_s.str() + "_ROB" + robID_s.str()+"_TimeBox";

    // used only if they have been set (controlled by the switch during filling)
    stringstream tdcID_s; tdcID_s << code.getTDC();
    stringstream chID_s; chID_s << code.getChannel();

    int index;
    switch (parameters.getUntrackedParameter<int>("TBhistoGranularity",1)) {
    case 1: // ROB
      index = code.getROBID();
      break;
    case 2: // TDC
      index = code.getTDCID();
      histoName = "FED" + dduID_s.str() 
	+ "_ROS" + rosID_s.str() 
	+ "_ROB" + robID_s.str()
	+ "_TDC" + tdcID_s.str() + "_TimeBox";
      break;
    case 3: // Ch
      index = code.getChannelID();
      histoName = "FED" + dduID_s.str() 
	+ "_ROS" + rosID_s.str() 
	+ "_ROB" + robID_s.str()
	+ "_TDC" + tdcID_s.str() 
	+ "_Channel" + chID_s.str() + "_TimeBox";
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
  

  if ( folder == "TDCError") {

    dbe->setCurrentFolder("DT/DataIntegrity/FED" + dduID_s.str()+"/ROS"+rosID_s.str()+"/ROB"+robID_s.str());

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
    dbe->setCurrentFolder("DT/DataIntegrity/FED" + dduID_s.str() + "/" + folder + rosID_s.str());

    // the SC histos belong to the ROS map (pay attention) since the data come from the corresponding ROS

    histoType = "SCTriggerBX";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_SCTriggerBX";
    string histoTitle = histoName + " (station vs BX)";
    (rosHistos[histoType])[code.getSCID()] = dbe->book2D(histoName,histoTitle,128,0,128,4,1,5);
    //    (rosHistos[histoType])[code.getSCID()] = dbe->book2D(histoName,histoTitle,128,0,128,5,0,5);

    //    (robHistos[histoType])[code.getROBID()] = dbe->book2D(histoName,histoTitle,3,0,3,4,0,4);

    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(1,"MB1",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(2,"MB2",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(3,"MB3",2);
    ((rosHistos[histoType])[code.getSCID()]) ->setBinLabel(4,"MB4",2);


    histoType = "SCTriggerQuality";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_SCTriggerQuality";
    histoTitle = histoName + "(quality vs station)";
    //    (rosHistos[histoType])[code.getSCID()] = dbe->book2D(histoName,histoTitle,5,0,5,8,0,8);
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


void DTDataIntegrityTask::processROS25(DTROS25Data & data, int ddu, int ros) {
  
  neventsROS25++;
  if ((neventsROS25%1000 == 0) &&debug)
    cout<<"[DTDataIntegrityTask]: "<<neventsROS25<<" events analyzed by processROS25"<<endl;
  
  DTROChainCoding code;
  code.setDDU(ddu);
  code.setROS(ros);

  string histoType;

  /// ROS Data
  histoType = "ROSSummary";
  if (rosSHistos[histoType].find(code.getDDUID()) == rosSHistos[histoType].end() ) {
    bookHistos( string("ROS_S"), code);
  }

  if (data.getROSTrailer().TPX() != 0) {
    if (debug) cout << " TXP error en ROS " << code.getROS() << endl;
    (rosSHistos.find("ROSSummary")->second).find(code.getDDUID())->second->Fill(9,code.getROS());
  }


  /// ROS Trailer
  histoType = "ROSTrailerBits";

  /// FIXME: EC* data are not correctly treated. One histo each is needed
  if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()) {
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(1,data.getROSTrailer().TFF());
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(2,data.getROSTrailer().TPX());
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(3,data.getROSTrailer().ECHO());
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(3,data.getROSTrailer().ECLO());
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(3,data.getROSTrailer().BCO());
  }
  else {
    bookHistos( string("ROS"), code);
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(1,data.getROSTrailer().TFF());
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(2,data.getROSTrailer().TPX());
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(3,data.getROSTrailer().ECHO());
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(3,data.getROSTrailer().ECLO());
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(3,data.getROSTrailer().BCO());
  }
  
  histoType = "ROSEventLenght";
  if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()) {
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(data.getROSTrailer().EventWordCount());
  }
  else {
    bookHistos( string("ROS"), code);
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(data.getROSTrailer().EventWordCount());
  }


// ROS errors


  if (neventsROS25 == 1) FirstRos = code.getROSID();
  if (code.getROSID() == FirstRos) nevents++ ;

  histoType = "ROSError";
  for (vector<DTROSErrorWord>::const_iterator error_it = data.getROSErrors().begin();
       error_it != data.getROSErrors().end(); error_it++) {

    if (debug)
      cout << " Error in ROS " << code.getROS() << " ROB Id " << (*error_it).robID() << " Error type " << (*error_it).errorType() << endl;

    (rosSHistos.find("ROSSummary")->second).find(code.getDDUID())->second->Fill((*error_it).errorType(),code.getROS());

    if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()) {
      //    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*error_it).robID(),
      //             (*error_it).errorType());
      if ((*error_it).errorType() != 4)
        (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*error_it).errorType(),(*error_it).robID());
    }
    else {
      bookHistos( string("ROS"), code);
      //    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*error_it).robID(),
      //             (*error_it).errorType());
      if ((*error_it).errorType() != 4)
        (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*error_it).errorType(),(*error_it).robID());
    }
  }

  int ROSDebug_BunchNumber;
  int ROSDebug_BcntResCntLow = 0;
  int ROSDebug_BcntResCntHigh = 0;
  int ROSDebug_BcntResCnt = 0;

  for (vector<DTROSDebugWord>::const_iterator debug_it = data.getROSDebugs().begin();
       debug_it != data.getROSDebugs().end(); debug_it++) {

    if ((*debug_it).debugType() == 0 ) {
      ROSDebug_BunchNumber = (*debug_it).debugMessage();
      histoType = "ROSDebug_BunchNumber";
      if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end())
        (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*debug_it).debugMessage());
      else {
        bookHistos( string("ROS"), code);
        (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill((*debug_it).debugMessage());
      }
    }

    if ((*debug_it).debugType() == 1 ) {
      ROSDebug_BcntResCntLow = (*debug_it).debugMessage();
      // This histo has been removed; 
      histoType = "ROSDebug_BcntResCntLow";
    }

    if ((*debug_it).debugType() == 2 ) {
      ROSDebug_BcntResCntHigh = (*debug_it).debugMessage();
      // This histo has been removed; 
      histoType = "ROSDebug_BcntResCntHigh";
    }
  }


  ROSDebug_BcntResCnt = (ROSDebug_BcntResCntHigh << 15) + ROSDebug_BcntResCntLow;
  if (debug)
    cout << " ROS: " << code.getROS() << " ROSDebug_BunchNumber " << ROSDebug_BunchNumber
	 << " ROSDebug_BcntResCnt " << ROSDebug_BcntResCnt << endl;

  histoType = "ROSDebug_BcntResCnt";
  if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end())
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(ROSDebug_BcntResCnt);
  else {
    bookHistos( string("ROS"), code);
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(ROSDebug_BcntResCnt);
  }


   ///	 Event words vs time
 	
  ROSWords_t(ResetCount_unfolded,code.getROS(),ROSDebug_BcntResCnt,nevents);

// fill hists it here
  histoType = "Event_word_vs_time";  	  
  if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()){
  (rosHistos.find(histoType)->second).find(code.getROSID())->second->
  		Fill((ResetCount_unfolded),data.getROSTrailer().EventWordCount());
  (rosHistos.find(histoType)->second).find(code.getROSID())->second->setAxisTitle("Time(s)",1);
   }
  else {
     (rosHistos.find(histoType)->second).find(code.getROSID())->second->
    		Fill((ResetCount_unfolded),data.getROSTrailer().EventWordCount());}  


	
  /// Trigger frequency
  
  
 frequency = 0;
 ROS_L1A_Frequency(code.getROS(),ROSDebug_BcntResCnt,neventsROS25,frequency,trigger_counter);

  histoType = "Trigger_frequency";
   if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()){
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(frequency);
        }
   else {
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(frequency);
        }

		
  /// ROB Group Header

  for (vector<DTROBHeader>::const_iterator rob_it = data.getROBHeaders().begin();
       rob_it != data.getROBHeaders().end(); rob_it++){

    code.setROB((*rob_it).first);
    DTROBHeaderWord robheader = (*rob_it).second;  

    histoType = "Bunch_ID";
    if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end())
    (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(robheader.bunchID());
    else (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(robheader.bunchID());
	        
    if (robheader.bunchID() != ROSDebug_BunchNumber) {
      //     fill ROS Summary plot
      (rosSHistos.find("ROSSummary")->second).find(code.getDDUID())->second->Fill(8,code.getROS());
      //     fill ROB Summary plot for that particular ROS
      histoType = "ROSError";
      if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()) //CB getROS->getROSID
	(rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(8,robheader.robID());
      else {
	bookHistos( string("ROS"), code);
	(rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(8,robheader.robID());//CB getROS->getROSID
	 }
	}
       }


/// ROB Trailer

  for (vector<DTROBTrailerWord>::const_iterator robt_it = data.getROBTrailers().begin();
       robt_it != data.getROBTrailers().end(); robt_it++) {       
   
   histoType = "ROB_mean";
   if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()) {
       (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(code.getROB(),
      								(*robt_it).wordCount());
       (rosHistos.find(histoType)->second).find(code.getROSID())->second->setAxisTitle("ROB #",1);							
       (rosHistos.find(histoType)->second).find(code.getROSID())->second->setAxisTitle("ROB wordcounts",2);
    }
   }
   
  /// TDC Data  
  for (vector<DTTDCData>::const_iterator tdc_it = data.getTDCData().begin();
       tdc_it != data.getTDCData().end(); tdc_it++) {


    DTTDCMeasurementWord tdcDatum = (*tdc_it).second;

    if ( tdcDatum.PC() !=0)  {
      if (debug) cout << " PC error en ROS " << code.getROS() << " TDC " << (*tdc_it).first << endl;
      //     fill ROS Summary plot
      (rosSHistos.find("ROSSummary")->second).find(code.getDDUID())->second->Fill(7,code.getROS());
      //     fill ROB Summary plot for that particular ROS
      histoType = "ROSError";
      if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end())
	(rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(7,(*tdc_it).first);
      else {
	bookHistos( string("ROS"), code);
	(rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(7,(*tdc_it).first);
      }
    }

    int index;
    switch (parameters.getUntrackedParameter<int>("TBhistoGranularity",1)) {
    case 1:
      code.setROB((*tdc_it).first);
      index = code.getROBID();
      break;
    case 2:
      code.setROB((*tdc_it).first);
      code.setTDC(tdcDatum.tdcID());
      index = code.getTDCID();
      break;
    case 3:
      code.setROB((*tdc_it).first);
      code.setTDC(tdcDatum.tdcID());
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


  /// TDC Error  
  for (vector<DTTDCError>::const_iterator tdc_it = data.getTDCError().begin();
       tdc_it != data.getTDCError().end(); tdc_it++) {

    code.setROB((*tdc_it).first);

    float type_TDC_error_for_plot_1 = 0;
    float type_TDC_error_for_plot_2 = 0;

    if ( ((*tdc_it).second).tdcError() & 0x4000 ) {
      if (debug)
	cout << " ROS " << code.getROS() << " ROB " << code.getROB() << " Internal fatal Error 4000 in TDC " << (*tdc_it).first << endl;
      type_TDC_error_for_plot_1 = 10;
      type_TDC_error_for_plot_2 = 0;
    }
    else if ( ((*tdc_it).second).tdcError() & 0x1b6d ) {
      if (debug)
	cout << " ROS " << code.getROS() << " ROB " << code.getROB() << " TDC FIFO full in TDC " << (*tdc_it).first << endl;
      type_TDC_error_for_plot_1 = 11;
      type_TDC_error_for_plot_2 = 1;
    }
    else if ( ((*tdc_it).second).tdcError() & 0x2492 ) {
      if (debug)
	cout << " ROS " << code.getROS() << " ROB " << code.getROB() << " L1 buffer overflow in TDC " << (*tdc_it).first << endl;
      type_TDC_error_for_plot_1 = 12;
      type_TDC_error_for_plot_2 = 2;
    }
    else {
      cout << " TDC error code not known " << ((*tdc_it).second).tdcError() << endl;
    }

    histoType = "ROSError";
    if (rosHistos[histoType].find(code.getROSID()) != rosHistos[histoType].end()) {
      (rosSHistos.find("ROSSummary")->second).find(code.getDDUID())->second->Fill(type_TDC_error_for_plot_1,code.getROS());
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(type_TDC_error_for_plot_1,(*tdc_it).first);
    }
    else {
      bookHistos( string("ROS"), code);
      (rosSHistos.find("ROSSummary")->second).find(code.getDDUID())->second->Fill(type_TDC_error_for_plot_1,code.getROS());
      (rosHistos.find(histoType)->second).find(code.getROSID())->second->Fill(type_TDC_error_for_plot_1,(*tdc_it).first);
    }

    histoType = "TDCError";
    if (robHistos[histoType].find(code.getROBID()) != robHistos[histoType].end()) {
      (robHistos.find(histoType)->second).find(code.getROBID())->second->Fill(type_TDC_error_for_plot_2,((*tdc_it).second).tdcID());
    }
    else {
      bookHistos( string("TDCError"), code);
      (robHistos.find(histoType)->second).find(code.getROBID())->second->Fill(type_TDC_error_for_plot_2,((*tdc_it).second).tdcID());
    }

  }


  if ( parameters.getUntrackedParameter<bool>("getSCInfo", false) ) {

    /// SC Data
    int stationGroup = 0 ; //= ((*sc_it).second)%2;
    for (vector<DTSectorCollectorData>::const_iterator sc_it = data.getSCData().begin();
	 sc_it != data.getSCData().end(); sc_it++) {

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

	histoType = "SCTriggerBX";
	if (rosHistos[histoType].find(code.getSCID()) != rosHistos[histoType].end())
	  (rosHistos.find(histoType)->second).find(code.getSCID())->second->Fill((*sc_it).second, 1+stationGroup*2);
	else {									       
	  bookHistos( string("SC"), code);
	  (rosHistos.find(histoType)->second).find(code.getSCID())->second->Fill((*sc_it).second, 1+stationGroup*2);
	}										       

	histoType = "SCTriggerQuality";						       
	if (rosHistos[histoType].find(code.getSCID()) != rosHistos[histoType].end())      
	  (rosHistos.find(histoType)->second).find(code.getSCID())->second->Fill(1+stationGroup*2,quality_LSB);
	else {									       
	  bookHistos( string("SC"), code);						       
	  (rosHistos.find(histoType)->second).find(code.getSCID())->second->Fill(1+stationGroup*2,quality_LSB);
	}
      }
    
      if (hasTrigger_MSB) {

	histoType = "SCTriggerBX";
	if (rosHistos[histoType].find(code.getSCID()) != rosHistos[histoType].end())
	  (rosHistos.find(histoType)->second).find(code.getSCID())->second->Fill((*sc_it).second, 2+stationGroup*2);
	else {									       
	  bookHistos( string("SC"), code);	
	  (rosHistos.find(histoType)->second).find(code.getSCID())->second->Fill((*sc_it).second, 2+stationGroup*2);
	}										       
      
	histoType = "SCTriggerQuality";						       
	if (rosHistos[histoType].find(code.getSCID()) != rosHistos[histoType].end())      
	  (rosHistos.find(histoType)->second).find(code.getSCID())->second->Fill(2+stationGroup*2,quality_MSB);
	else {									       
	  bookHistos( string("SC"), code);						       
	  (rosHistos.find(histoType)->second).find(code.getSCID())->second->Fill(2+stationGroup*2,quality_MSB);
	}
      }
      stationGroup = (stationGroup == 0 ? 1 : 0);  //switch between MB1-2 and MB3-4 data
    }

  }   

}

void DTDataIntegrityTask::processFED(DTDDUData & data, const std::vector<DTROS25Data> & rosData, int ddu) {

  neventsDDU++;
  if ((neventsDDU%1000 == 0) && debug)
    cout<<"[DTDataIntegrityTask]: "<<neventsDDU<<" events analyzed by processFED"<<endl;

  DTROChainCoding code;
  code.setDDU(ddu);

  string histoType;

  FEDTrailer trailer = data.getDDUTrailer();
  FEDHeader header = data.getDDUHeader();
  DTDDUSecondStatusWord secondWord = data.getSecondStatusWord();

  //1D HISTO WITH TTS VALUES form trailer (7 bins = 7 values)
  histoType = "TTSValues";
  if (dduHistos[histoType].find(code.getDDUID()) == dduHistos[histoType].end()) {
    bookHistos( string("DDU"), code);
  }
  
  switch(trailer.ttsBits()){
  case 0:{ //disconnected
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(0);
    break;
  }
  case 1:{ //warning overflow
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(1);
    break;
  }
  case 2:{ //out of sinch
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(2);
    break;
  }
  case 4:{ //busy
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(3);
    break;
  }
  case 8:{ //ready
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(4);
    break;
  }
  case 12:{ //error
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(5);
    break;
  }
  case 16:{ //disconnected
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(6);
    break;
  }
  default:{
    cout<<"[DTDataInetegrityTask] DDU control: wrong TTS value "<<trailer.ttsBits()<<endl;
  }
  }
  
  //1D HISTO: IF TTS=2,12 CHECK L1A AND BX MISIMATCH, FIFO AND ROS ERROR (from status words)
  if(trailer.ttsBits()==2){
    histoType = "TTS_2";
  }
  if(trailer.ttsBits()==12){
    histoType = "TTS_12";
  }

  if(trailer.ttsBits()==2 || trailer.ttsBits()==12){
    if (dduHistos[histoType].find(code.getDDUID()) == dduHistos[histoType].end()) {
      bookHistos( string("DDU"), code);
    }
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(0,secondWord.l1AIDError());
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(1,secondWord.bxIDError());
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(2,(secondWord.fifoFull() & 0x1));
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(3,(secondWord.fifoFull() & 0x2));
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(4,(secondWord.fifoFull() & 0x4));
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(5,(secondWord.inputFifoFull() & 0x1));
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(6,(secondWord.inputFifoFull() & 0x2));
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(7,(secondWord.inputFifoFull() & 0x4));
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(8,secondWord.outputFifoFull());
    int channel=1;
    for (vector<DTDDUFirstStatusWord>::const_iterator fsw_it = data.getFirstStatusWord().begin();
	 fsw_it != data.getFirstStatusWord().end(); fsw_it++) {
      if((*fsw_it).timeout() || (*fsw_it).eventTrailerLost() || (*fsw_it).opticalFiberSignalLost() ||
	 (*fsw_it).tlkPropagationError()||(*fsw_it).tlkPatternError() ||(*fsw_it).tlkSignalLost() || (*fsw_it).errorFromROS())
	(dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(8+channel,1);
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
  histoType = "EventLenght";
  if (dduHistos[histoType].find(code.getDDUID()) == dduHistos[histoType].end()) {
    bookHistos( string("DDU"), code);
  }
  (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(trailer.lenght());

  //1D HISTO: EVENT TYPE from header
  //cout<<"1D HISTO WITH EVENT TYPE from header"<<endl;
  histoType = "EventType";
  if (dduHistos[histoType].find(code.getDDUID()) == dduHistos[histoType].end()) {
    bookHistos( string("DDU"), code);
  }
  (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(header.triggerType());  

  //1D HISTO: NUMBER OF ROS IN THE EVENTS from 2nd status word
  int rosList = secondWord.rosList();
  vector<int> rosPositions;
  for(int i=0;i<12;i++){
    if(rosList & 0x1)
      rosPositions.push_back(i);
    rosList >>= 1;
  }

  histoType = "ROSList";   
  if (dduHistos[histoType].find(code.getDDUID()) == dduHistos[histoType].end()) {
    bookHistos( string("DDU"), code);
  } 
  (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(rosPositions.size());

  //2D HISTO: ROS VS STATUS (8 BIT = 8 BIN) from 1st-2nd status words (9th BIN FROM LIST OF ROS in 2nd status word)
  histoType = "ROSStatus";   
  if (dduHistos[histoType].find(code.getDDUID()) == dduHistos[histoType].end()) {
    bookHistos( string("DDU"), code);
  } 
  int channel=0;
  for (vector<DTDDUFirstStatusWord>::const_iterator fsw_it = data.getFirstStatusWord().begin();
       fsw_it != data.getFirstStatusWord().end(); fsw_it++) {
    // assuming association one-to-one between DDU channel and ROS
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(0,channel,(*fsw_it).channelEnabled());
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(1,channel,(*fsw_it).timeout());
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(2,channel,(*fsw_it).eventTrailerLost());
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(3,channel,(*fsw_it).opticalFiberSignalLost());
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(4,channel,(*fsw_it).tlkPropagationError());
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(5,channel,(*fsw_it).tlkPatternError());
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(6,channel,(*fsw_it).tlkSignalLost());
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(7,channel,(*fsw_it).errorFromROS());
    channel++;
  }
  //9th BIN FROM LIST OF ROS in 2nd status word
  for(vector<int>::const_iterator channel_it = rosPositions.begin(); channel_it != rosPositions.end(); channel_it++){
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(8,(*channel_it),1);
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
  histoType = "FIFOStatus";   
  if (dduHistos[histoType].find(code.getDDUID()) == dduHistos[histoType].end()) {
    bookHistos( string("DDU"), code);
  } 
  
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
      (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(i,0);
    }
    if(inputFifoAlmostFull & 0x1){
      fifoStatus[i]=1;
      (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(i,1);
    }
    if(fifoFull & 0x1){
      fifoStatus[3+i]=0;
      (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(3+i,0);
    }
    if(fifoAlmostFull & 0x1){
      fifoStatus[3+i]=1;
      (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(3+i,1);
    }
    if(!(inputFifoFull & 0x1) && !(inputFifoAlmostFull & 0x1)){
      fifoStatus[i]=2;
      (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(i,2);
    }
    if(!(fifoFull & 0x1) && !(fifoAlmostFull & 0x1)){
      fifoStatus[3+i]=2;
      (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(3+i,2);
    }
    inputFifoFull >>= 1;
    inputFifoAlmostFull >>= 1;
    fifoFull >>= 1;
    fifoAlmostFull >>= 1;
  }

  if(outputFifoFull){
    fifoStatus[6]=0;
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(6,0);
  }
  if(outputFifoAlmostFull){
    fifoStatus[6]=1;
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(6,1);
  }
  if(!outputFifoFull && !outputFifoAlmostFull){
    fifoStatus[6]=2;
    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill(6,2);
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


  if(trailer.ttsBits()==2){   //DDU OUT OF SYNCH

    //If BX_ID error identify which ROS has wrong BX
    histoType = "BX_IDErrorROS";
    if (dduHistos[histoType].find(code.getDDUID()) == dduHistos[histoType].end()) {
      bookHistos( string("DDU"), code);
    } 
    for (vector<DTROS25Data>::const_iterator ros_it = rosData.begin();
	 ros_it != rosData.end(); ros_it++) {
      for (vector<DTROSDebugWord>::const_iterator debug_it = (*ros_it).getROSDebugs().begin();
	   debug_it != (*ros_it).getROSDebugs().end(); debug_it++) {
	if ((*debug_it).debugType() == 0 ) {
	  int ROSDebug_BXID = (*debug_it).debugMessage();
	  if(ROSDebug_BXID != header.bxID()){
	    (dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill((*ros_it).getROSID()-1);
	    //FIXME: how to notify this error in a log file
	    cout << "BX_ID error from ROS "<<(*ros_it).getROSID()<<" :"
		 <<" ROSDebug_BXID "<< ROSDebug_BXID
		 <<"   DDUHeader_BXID "<< header.bxID()<<endl;
	  }
	}
      }
    }

    //If L1A_ID error identify which ROS has wrong L1A 
    histoType = "L1A_IDErrorROS";
    if (dduHistos[histoType].find(code.getDDUID()) == dduHistos[histoType].end()) {
      bookHistos( string("DDU"), code);
    } 
    for (vector<DTROS25Data>::const_iterator ros_it = rosData.begin();
	 ros_it != rosData.end(); ros_it++) {
      int ROSHeader_TTCCount = ((*ros_it).getROSHeader()).TTCEventCounter();
      if(ROSHeader_TTCCount != header.lvl1ID()-1){
	(dduHistos.find(histoType)->second).find(code.getDDUID())->second->Fill((*ros_it).getROSID()-1);
	//FIXME: how to notify this error in a log file
	cout << "L1A_ID error from ROS "<<(*ros_it).getROSID()<<" :"
	     <<" ROSHeader_TTCeventcounter " << ROSHeader_TTCCount
	     <<"   DDUHeader_lvl1ID "<< header.lvl1ID()<<endl;
      }
    }
  }
}

  
