/*
 * \file DTDataIntegrityTask.cc
 *
 * \author M. Zanetti (INFN Padova), S. Bolognesi (INFN Torino), G. Cerminara (INFN Torino)
 *
 */

#include "DQM/DTMonitorModule/interface/DTDataIntegrityTask.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DTDigi/interface/DTControlData.h"
#include "DataFormats/DTDigi/interface/DTDDUWords.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQM/DTMonitorModule/interface/DTTimeEvolutionHisto.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <math.h>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace edm;

DTDataIntegrityTask::DTDataIntegrityTask(const edm::ParameterSet& ps) : nevents(0) {

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
  << "[DTDataIntegrityTask]: Constructor" <<endl;

  dduToken = consumes<DTDDUCollection>(ps.getParameter<InputTag>("dtDDULabel"));
  ros25Token = consumes<DTROS25Collection>(ps.getParameter<InputTag>("dtROS25Label"));

  neventsDDU = 0;
  neventsROS25 = 0;

  FEDIDmin = FEDNumbering::MINDTFEDID;
  FEDIDMax = FEDNumbering::MAXDTFEDID;

//   If you want info VS time histos
//   doTimeHisto =  ps.getUntrackedParameter<bool>("doTimeHisto", false);
//   Plot quantities about SC
  getSCInfo = ps.getUntrackedParameter<bool>("getSCInfo", false);

  fedIntegrityFolder    = ps.getUntrackedParameter<string>("fedIntegrityFolder","DT/FEDIntegrity");

  string processingMode = ps.getUntrackedParameter<string>("processingMode","Online");

  // processing mode flag to select plots to be produced and basedirs CB vedi se farlo meglio...
  if (processingMode == "Online") {
    mode = 0;
  } else if(processingMode == "SM") {
    mode = 1;
  } else if (processingMode == "Offline") {
    mode = 2;
  } else if (processingMode == "HLT") {
    mode = 3;
  } else {
    throw cms::Exception("MissingParameter")
      << "[DTDataIntegrityTask]: processingMode :" << processingMode
      << " invalid! Must be Online, SM, Offline or HLT !" << endl;
  }

}


DTDataIntegrityTask::~DTDataIntegrityTask() {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    <<"[DTDataIntegrityTask]: Destructor. Analyzed "<< neventsDDU <<" events"<<endl;
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << "[DTDataIntegrityTask]: postEndJob called!" <<endl;
}

/*
  Folder Structure:
  - One folder for each DDU, named FEDn
  - Inside each DDU folder the DDU histos and the ROSn folder
  - Inside each ROS folder the ROS histos and the ROBn folder
  - Inside each ROB folder one occupancy plot and the TimeBoxes
  with the chosen granularity (simply change the histo name)
*/

void DTDataIntegrityTask::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & iSetup) {

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: postBeginJob" <<endl;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask] Get DQMStore service" << endl;

  // Loop over the DT FEDs

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << " FEDS: " << FEDIDmin  << " to " <<  FEDIDMax << " in the RO" << endl;

  // book FED integrity histos
  bookHistos(ibooker, FEDIDmin, FEDIDMax);

  // static booking of the histograms
  for(int fed = FEDIDmin; fed <= FEDIDMax; ++fed) { // loop over the FEDs in the readout
    DTROChainCoding code;
    code.setDDU(fed);

    bookHistos(ibooker, string("ROS_S"), code);

    bookHistos(ibooker, string("DDU"), code);

    for(int ros = 1; ros <= 12; ++ros) {// loop over all ROS
      code.setROS(ros);
      bookHistosROS25(ibooker, code);
    }
  }
}

void DTDataIntegrityTask::bookHistos(DQMStore::IBooker & ibooker, const int fedMin, const int fedMax) {

  ibooker.setCurrentFolder("DT/EventInfo/Counters");
  nEventMonitor = ibooker.bookFloat("nProcessedEventsDataIntegrity");

  // Standard FED integrity histos
  ibooker.setCurrentFolder(topFolder(true));

  int nFED = (fedMax - fedMin)+1;

  hFEDEntry = ibooker.book1D("FEDEntries","# entries per DT FED",nFED,fedMin,fedMax+1);
  hFEDFatal = ibooker.book1D("FEDFatal","# fatal errors DT FED",nFED,fedMin,fedMax+1);
  hFEDNonFatal = ibooker.book1D("FEDNonFatal","# NON fatal errors DT FED",nFED,fedMin,fedMax+1);


  ibooker.setCurrentFolder(topFolder(false));
  hTTSSummary = ibooker.book2D("TTSSummary","Summary Status TTS",nFED,fedMin,fedMax+1,9,1,10);
  hTTSSummary->setAxisTitle("FED",1);
  hTTSSummary->setBinLabel(1,"ROS PAF",2);
  hTTSSummary->setBinLabel(2,"DDU PAF",2);
  hTTSSummary->setBinLabel(3,"ROS PAF",2);
  hTTSSummary->setBinLabel(4,"DDU PAF",2);
  hTTSSummary->setBinLabel(5,"DDU Full",2);
  hTTSSummary->setBinLabel(6,"L1A Mism.",2);
  hTTSSummary->setBinLabel(7,"ROS Error",2);
  hTTSSummary->setBinLabel(8,"BX Mism.",2);
  hTTSSummary->setBinLabel(9,"DDU Logic Err.",2);

  // bookkeeping of the

  hCorruptionSummary =  ibooker.book2D("DataCorruptionSummary", "Data Corruption Sources",
				   nFED,fedMin,fedMax+1, 8, 1, 9);
  hCorruptionSummary->setAxisTitle("FED",1);
  hCorruptionSummary->setBinLabel(1,"Miss Ch.",2);
  hCorruptionSummary->setBinLabel(2,"ROS BX mism",2);
  hCorruptionSummary->setBinLabel(3,"DDU BX mism",2);
  hCorruptionSummary->setBinLabel(4,"ROS L1A mism",2);
  hCorruptionSummary->setBinLabel(5,"Miss Payload",2);
  hCorruptionSummary->setBinLabel(6,"FCRC bit",2);
  hCorruptionSummary->setBinLabel(7,"Header check",2);
  hCorruptionSummary->setBinLabel(8,"Triler Check",2);

}



void DTDataIntegrityTask::bookHistos(DQMStore::IBooker & ibooker, string folder, DTROChainCoding code) {

  stringstream dduID_s; dduID_s << code.getDDU();
  stringstream rosID_s; rosID_s << code.getROS();
  stringstream robID_s; robID_s << code.getROB();
  int wheel = (code.getDDUID() - 770)%5 - 2;
  stringstream wheel_s; wheel_s << wheel;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << " Booking histos for FED: " << code.getDDU() << " ROS: " << code.getROS()
    << " ROB: " << code.getROB() << " folder: " << folder << endl;

  string histoType;
  string histoName;
  string histoTitle;
  MonitorElement* histo = 0;

  // DDU Histograms
  if (folder == "DDU") {

    ibooker.setCurrentFolder(topFolder(false) + "FED" + dduID_s.str());

    histoType = "EventLenght";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    histoTitle = "Event Lenght (Bytes) FED " +  dduID_s.str();
    (dduHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoTitle,501,0,16032);

    if(mode > 2) return;

    histoType = "ROSStatus";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = ibooker.book2D(histoName,histoName,12,0,12,12,0,12);
    histo = (dduHistos[histoType])[code.getDDUID()];
    histo->setBinLabel(1,"ch.enabled",1);
    histo->setBinLabel(2,"timeout",1);
    histo->setBinLabel(3,"ev.trailer lost",1);
    histo->setBinLabel(4,"opt.fiber lost",1);
    histo->setBinLabel(5,"tlk.prop.error",1);
    histo->setBinLabel(6,"tlk.pattern error",1);
    histo->setBinLabel(7,"tlk.sign.lost",1);
    histo->setBinLabel(8,"error from ROS",1);
    histo->setBinLabel(9,"if ROS in events",1);
    histo->setBinLabel(10,"Miss. Evt.",1);
    histo->setBinLabel(11,"Evt. ID Mismatch",1);
    histo->setBinLabel(12,"BX Mismatch",1);

    histo->setBinLabel(1,"ROS 1",2);
    histo->setBinLabel(2,"ROS 2",2);
    histo->setBinLabel(3,"ROS 3",2);
    histo->setBinLabel(4,"ROS 4",2);
    histo->setBinLabel(5,"ROS 5",2);
    histo->setBinLabel(6,"ROS 6",2);
    histo->setBinLabel(7,"ROS 7",2);
    histo->setBinLabel(8,"ROS 8",2);
    histo->setBinLabel(9,"ROS 9",2);
    histo->setBinLabel(10,"ROS 10",2);
    histo->setBinLabel(11,"ROS 11",2);
    histo->setBinLabel(12,"ROS 12",2);

    if(mode > 1) return;

    histoType = "FEDAvgEvLenghtvsLumi";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    histoTitle = "Avg Event Lenght (Bytes) vs LumiSec FED " +  dduID_s.str();
    dduTimeHistos[histoType][code.getDDUID()] = new DTTimeEvolutionHisto(ibooker,histoName,histoTitle,200,10,true,0);

    histoType = "TTSValues";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoName,8,0,8);
    histo = (dduHistos[histoType])[code.getDDUID()];
    histo->setBinLabel(1,"disconnected",1);
    histo->setBinLabel(2,"warning overflow",1);
    histo->setBinLabel(3,"out of synch",1);
    histo->setBinLabel(4,"busy",1);
    histo->setBinLabel(5,"ready",1);
    histo->setBinLabel(6,"error",1);
    histo->setBinLabel(7,"disconnected",1);
    histo->setBinLabel(8,"unknown",1);

    histoType = "EventType";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoName,2,1,3);
    histo = (dduHistos[histoType])[code.getDDUID()];
    histo->setBinLabel(1,"physics",1);
    histo->setBinLabel(2,"calibration",1);

    histoType = "ROSList";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    histoTitle = "# of ROS in the FED payload (FED" + dduID_s.str() + ")";
    (dduHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoTitle,13,0,13);

    histoType = "FIFOStatus";
    histoName = "FED" + dduID_s.str() + "_" + histoType;
    (dduHistos[histoType])[code.getDDUID()] = ibooker.book2D(histoName,histoName,7,0,7,3,0,3);
    histo = (dduHistos[histoType])[code.getDDUID()];
    histo->setBinLabel(1,"Input ch1-4",1);
    histo->setBinLabel(2,"Input ch5-8",1);
    histo->setBinLabel(3,"Input ch9-12",1);
    histo->setBinLabel(4,"Error/L1A ch1-4",1);
    histo->setBinLabel(5,"Error/L1A ch5-8",1);
    histo->setBinLabel(6,"Error/L1A ch9-12",1);
    histo->setBinLabel(7,"Output",1);
    histo->setBinLabel(1,"Full",2);
    histo->setBinLabel(2,"Almost Full",2);
    histo->setBinLabel(3,"Not Full",2);

    histoType = "BXID";
    histoName = "FED" + dduID_s.str() + "_BXID";
    histoTitle = "Distrib. BX ID (FED" + dduID_s.str() + ")";
    (dduHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoTitle,3600,0,3600);

  }

  // ROS Histograms
  if ( folder == "ROS_S" ) { // The summary of the error of the ROS on the same FED
    ibooker.setCurrentFolder(topFolder(false));

    histoType = "ROSSummary";
    histoName = "FED" + dduID_s.str() + "_ROSSummary";
    string histoTitle = "Summary Wheel" + wheel_s.str() + " (FED " + dduID_s.str() + ")";

    ((rosSHistos[histoType])[code.getDDUID()]) = ibooker.book2D(histoName,histoTitle,20,0,20,12,1,13);
    MonitorElement *histo = ((rosSHistos[histoType])[code.getDDUID()]);
    // ROS error bins
    histo ->setBinLabel(1,"Link TimeOut",1);
    histo ->setBinLabel(2,"Ev.Id.Mis.",1);
    histo ->setBinLabel(3,"FIFO almost full",1);
    histo ->setBinLabel(4,"FIFO full",1);
    histo ->setBinLabel(5,"CEROS timeout",1);
    histo ->setBinLabel(6,"Max. wds",1);
    histo ->setBinLabel(7,"WO L1A FIFO",1);
    histo ->setBinLabel(8,"TDC parity err.",1);
    histo ->setBinLabel(9,"BX ID Mis.",1);
    histo ->setBinLabel(10,"TXP",1);
    histo ->setBinLabel(11,"L1A almost full",1);
    histo ->setBinLabel(12,"Ch. blocked",1);
    histo ->setBinLabel(13,"Ev. Id. Mis.",1);
    histo ->setBinLabel(14,"CEROS blocked",1);
    // TDC error bins
    histo ->setBinLabel(15,"TDC Fatal",1);
    histo ->setBinLabel(16,"TDC RO FIFO ov.",1);
    histo ->setBinLabel(17,"TDC L1 buf. ov.",1);
    histo ->setBinLabel(18,"TDC L1A FIFO ov.",1);
    histo ->setBinLabel(19,"TDC hit err.",1);
    histo ->setBinLabel(20,"TDC hit rej.",1);

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
    ibooker.setCurrentFolder(topFolder(false) + "FED" + dduID_s.str() + "/" + folder + rosID_s.str());


    histoType = "ROSError";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSError";
    histoTitle = histoName + " (ROBID error summary)";
    if(mode <= 1)
      (rosHistos[histoType])[code.getROSID()] = ibooker.book2D(histoName,histoTitle,17,0,17,26,0,26);
    else
      (rosHistos[histoType])[code.getROSID()] = ibooker.book2D(histoName,histoTitle,11,0,11,26,0,26);

    MonitorElement* histo = (rosHistos[histoType])[code.getROSID()];
    // ROS error bins
    histo->setBinLabel(1,"Link TimeOut",1);
    histo->setBinLabel(2,"Ev.Id.Mis.",1);
    histo->setBinLabel(3,"FIFO almost full",1);
    histo->setBinLabel(4,"FIFO full",1);
    histo->setBinLabel(5,"CEROS timeout",1);
    histo->setBinLabel(6,"Max. wds",1);
    histo->setBinLabel(7,"TDC parity err.",1);
    histo->setBinLabel(8,"BX ID Mis.",1);
    histo->setBinLabel(9,"Ch. blocked",1);
    histo->setBinLabel(10,"Ev. Id. Mis.",1);
    histo->setBinLabel(11,"CEROS blocked",1);
    if(mode <= 1) {
      // TDC error bins
      histo->setBinLabel(12,"TDC Fatal",1);
      histo->setBinLabel(13,"TDC RO FIFO ov.",1);
      histo->setBinLabel(14,"TDC L1 buf. ov.",1);
      histo->setBinLabel(15,"TDC L1A FIFO ov.",1);
      histo->setBinLabel(16,"TDC hit err.",1);
      histo->setBinLabel(17,"TDC hit rej.",1);
    }
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

    if(mode > 1) return;

    histoType = "ROSEventLenght";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_ROSEventLenght";
    histoTitle = "Event Lenght (Bytes) FED " +  dduID_s.str() + " ROS " + rosID_s.str();
    (rosHistos[histoType])[code.getROSID()] = ibooker.book1D(histoName,histoTitle,101,0,1616);

    histoType = "ROSAvgEventLenghtvsLumi";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + histoType;
    histoTitle = "Event Lenght (Bytes) FED " +  dduID_s.str() + " ROS " + rosID_s.str();
    rosTimeHistos[histoType][code.getROSID()] = new DTTimeEvolutionHisto(ibooker,histoName,histoTitle,200,10,true,0);

    histoType = "TDCError";
    histoName = "FED" + dduID_s.str() + "_" + folder + rosID_s.str() + "_TDCError";
    histoTitle = histoName + " (ROBID error summary)";
    (rosHistos[histoType])[code.getROSID()] = ibooker.book2D(histoName,histoTitle,24,0,24,25,0,25);
    histo = (rosHistos[histoType])[code.getROSID()];
    // TDC error bins
    histo->setBinLabel(1,"Fatal",1);
    histo->setBinLabel(2,"RO FIFO ov.",1);
    histo->setBinLabel(3,"L1 buf. ov.",1);
    histo->setBinLabel(4,"L1A FIFO ov.",1);
    histo->setBinLabel(5,"hit err.",1);
    histo->setBinLabel(6,"hit rej.",1);
    histo->setBinLabel(7,"Fatal",1);
    histo->setBinLabel(8,"RO FIFO ov.",1);
    histo->setBinLabel(9,"L1 buf. ov.",1);
    histo->setBinLabel(10,"L1A FIFO ov.",1);
    histo->setBinLabel(11,"hit err.",1);
    histo->setBinLabel(12,"hit rej.",1);
    histo->setBinLabel(13,"Fatal",1);
    histo->setBinLabel(14,"RO FIFO ov.",1);
    histo->setBinLabel(15,"L1 buf. ov.",1);
    histo->setBinLabel(16,"L1A FIFO ov.",1);
    histo->setBinLabel(17,"hit err.",1);
    histo->setBinLabel(18,"hit rej.",1);
    histo->setBinLabel(19,"Fatal",1);
    histo->setBinLabel(20,"RO FIFO ov.",1);
    histo->setBinLabel(21,"L1 buf. ov.",1);
    histo->setBinLabel(22,"L1A FIFO ov.",1);
    histo->setBinLabel(23,"hit err.",1);
    histo->setBinLabel(24,"hit rej.",1);

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

    histoType = "ROB_mean";
    histoName = "FED" + dduID_s.str() + "_" + "ROS" + rosID_s.str() + "_ROB_mean";
    string fullName = topFolder(false) + "FED" + dduID_s.str() + "/" + folder + rosID_s.str()+ "/" + histoName;
    names.insert (pair<std::string,std::string> (histoType,string(fullName)));
    (rosHistos[histoType])[code.getROSID()] = ibooker.book2D(histoName,histoName,25,0,25,100,0,100);
    (rosHistos[histoType])[code.getROSID()]->setAxisTitle("ROB #",1);
    (rosHistos[histoType])[code.getROSID()]->setAxisTitle("ROB wordcounts",2);

  }

  // SC Histograms
  if ( folder == "SC" ) {
    // The plots are per wheel
    ibooker.setCurrentFolder(topFolder(false) + "FED" + dduID_s.str());

    // SC data Size
    histoType = "SCSizeVsROSSize";
    histoName = "FED" + dduID_s.str() + "_SCSizeVsROSSize";
    histoTitle = "SC size vs SC (FED " + dduID_s.str() + ")";
    rosHistos[histoType][code.getSCID()] = ibooker.book2D(histoName,histoTitle,12,1,13,51,-1,50);
    rosHistos[histoType][code.getSCID()]->setAxisTitle("SC",1);

  }
}


void DTDataIntegrityTask::bookHistosROS25(DQMStore::IBooker & ibooker, DTROChainCoding code) {
  bookHistos(ibooker, string("ROS"), code);

    if(mode <= 1)
      if(getSCInfo)
	bookHistos(ibooker, string("SC"), code);
}


void DTDataIntegrityTask::processROS25(DTROS25Data & data, int ddu, int ros) {

  neventsROS25++; // FIXME: implement a counter which makes sense

      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< "[DTDataIntegrityTask]: " << neventsROS25 << " events analyzed by processROS25" << endl;

  // The ID of the RO board (used to map the histos)
  DTROChainCoding code;
  code.setDDU(ddu);
  code.setROS(ros);

  MonitorElement* ROSSummary = rosSHistos["ROSSummary"][code.getDDUID()];

  // Summary of all ROB errors
  MonitorElement* ROSError = 0;
  if(mode <= 2) ROSError = rosHistos["ROSError"][code.getROSID()];

  if ( (mode<=2) && (!ROSError) ) {
    LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") <<
	"Trying to access non existing ME at ROSID " << code.getROSID() <<
	std::endl;
    return;
  }

  // L1A ids to be checked against FED one
  rosL1AIdsPerFED[ddu].insert(data.getROSHeader().TTCEventCounter());

  // ROS errors


  // check for TPX errors
  if (data.getROSTrailer().TPX() != 0) {
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << " TXP error en ROS "
								      << code.getROS() << endl;
    ROSSummary->Fill(9,code.getROS());
  }

  // L1 Buffer almost full (non-critical error!)
  if(data.getROSTrailer().l1AFifoOccupancy() > 31) {
     ROSSummary->Fill(10,code.getROS());
   }


  for (vector<DTROSErrorWord>::const_iterator error_it = data.getROSErrors().begin();
       error_it != data.getROSErrors().end(); error_it++) { // Loop over ROS error words

    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << " Error in ROS " << code.getROS()
								      << " ROB Id " << (*error_it).robID()
								      << " Error type " << (*error_it).errorType() << endl;

    // Fill the ROSSummary (1 per FED) histo
    ROSSummary->Fill((*error_it).errorType(), code.getROS());
    if((*error_it).errorType() <= 11) { // set error flag
       eventErrorFlag = true;
    }

    if(mode <= 2) {
      // Fill the ROB Summary (1 per ROS) histo
      if ((*error_it).robID() != 31) {
	ROSError->Fill((*error_it).errorType(),(*error_it).robID());
      }
      else if ((*error_it).errorType() == 4) {
	vector<int> channelBins;
	channelsInROS((*error_it).cerosID(),channelBins);
	vector<int>::const_iterator channelIt  = channelBins.begin();
	vector<int>::const_iterator channelEnd = channelBins.end();
	for(;channelIt!=channelEnd;++channelIt) {
	  ROSError->Fill(4,(*channelIt));
	}
      }
    }
  }


  int ROSDebug_BunchNumber = -1;

  for (vector<DTROSDebugWord>::const_iterator debug_it = data.getROSDebugs().begin();
       debug_it != data.getROSDebugs().end(); debug_it++) { // Loop over ROS debug words

    int debugROSSummary = 0;
    int debugROSError   = 0;
    vector<int> debugBins;
    bool hasEvIdMis = false;
    vector<int> evIdMisBins;

    if ((*debug_it).debugType() == 0 ) {
      ROSDebug_BunchNumber = (*debug_it).debugMessage();
    } else if ((*debug_it).debugType() == 1 ) {
      // not used
      // ROSDebug_BcntResCntLow = (*debug_it).debugMessage();
    } else if ((*debug_it).debugType() == 2 ) {
      // not used
      // ROSDebug_BcntResCntHigh = (*debug_it).debugMessage();
    } else if ((*debug_it).debugType() == 3) {
      if ((*debug_it).dontRead()){
	debugROSSummary = 11;
	debugROSError   = 8;
	if (mode <= 2) channelsInCEROS((*debug_it).cerosIdCerosStatus(),(*debug_it).dontRead(),debugBins);
      } if ((*debug_it).evIdMis()){
	hasEvIdMis = true;
	if (mode <= 2) channelsInCEROS((*debug_it).cerosIdCerosStatus(),(*debug_it).evIdMis(),evIdMisBins);
      }
    } else if ((*debug_it).debugType() == 4 &&
	       (*debug_it).cerosIdRosStatus()){
      debugROSSummary = 13;
      debugROSError   = 10;
      if (mode <= 2) channelsInROS((*debug_it).cerosIdRosStatus(),debugBins);
    }

    if (debugROSSummary) {
      ROSSummary->Fill(debugROSSummary,code.getROS());
      if (mode <= 2) {
	vector<int>::const_iterator channelIt  = debugBins.begin();
	vector<int>::const_iterator channelEnd = debugBins.end();
	for (;channelIt!=channelEnd;++channelIt) {
	  ROSError->Fill(debugROSError,(*channelIt));
	}
      }
    }

    if (hasEvIdMis) {
      ROSSummary->Fill(12,code.getROS());
      if (mode <= 2) {
	vector<int>::const_iterator channelIt  = evIdMisBins.begin();
	vector<int>::const_iterator channelEnd = evIdMisBins.end();
	for (;channelIt!=channelEnd;++channelIt) {
	  ROSError->Fill(9,(*channelIt));
	}
      }
    }

  }

  // ROB Group Header
  // Check the BX of the ROB headers against the BX of the ROS
  for (vector<DTROBHeader>::const_iterator rob_it = data.getROBHeaders().begin();
       rob_it != data.getROBHeaders().end(); rob_it++) { // loop over ROB headers

    code.setROB((*rob_it).first);
    DTROBHeaderWord robheader = (*rob_it).second;

    rosBxIdsPerFED[ddu].insert(ROSDebug_BunchNumber);

    if (robheader.bunchID() != ROSDebug_BunchNumber) {
      // fill ROS Summary plot
      ROSSummary->Fill(8,code.getROS());
      eventErrorFlag = true;

      // fill ROB Summary plot for that particular ROS
      if(mode <= 2) ROSError->Fill(7,robheader.robID());
    }
  }


  if(mode <= 1) { // produce only when not in HLT
    // ROB Trailer
    for (vector<DTROBTrailerWord>::const_iterator robt_it = data.getROBTrailers().begin();
	 robt_it != data.getROBTrailers().end(); robt_it++) { // loop over ROB trailers
      float  wCount = (*robt_it).wordCount()<100. ? (*robt_it).wordCount() : 99.9;
      rosHistos["ROB_mean"][code.getROSID()]->Fill((*robt_it).robID(),wCount);
    }

    // Plot the event lenght //NOHLT
    int rosEventLenght = data.getROSTrailer().EventWordCount()*4;
    rosTimeHistos["ROSAvgEventLenghtvsLumi"][code.getROSID()]->accumulateValueTimeSlot(rosEventLenght);

    if(rosEventLenght > 1600) rosEventLenght = 1600;
    rosHistos["ROSEventLenght"][code.getROSID()]->Fill(rosEventLenght);
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
      if(mode <= 2) ROSError->Fill(6,(*tdc_it).first);
    }
  }

  // TDC Error
  for (vector<DTTDCError>::const_iterator tdc_it = data.getTDCError().begin();
       tdc_it != data.getTDCError().end(); tdc_it++) { // loop over TDC errors

    code.setROB((*tdc_it).first);

    int tdcError_ROSSummary = 0;
    int tdcError_ROSError = 0;
    int tdcError_TDCHisto = 0;

    if(((*tdc_it).second).tdcError() & 0x4000 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " ROS " << code.getROS() << " ROB " << code.getROB()
	<< " Internal fatal Error 4000 in TDC " << (*tdc_it).first << endl;

      tdcError_ROSSummary = 14;
      tdcError_ROSError   = 11;
      tdcError_TDCHisto   = 0;

    } else if ( ((*tdc_it).second).tdcError() & 0x0249 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " ROS " << code.getROS() << " ROB " << code.getROB()
	<< " TDC FIFO overflow in TDC " << (*tdc_it).first << endl;

      tdcError_ROSSummary = 15;
      tdcError_ROSError   = 12;
      tdcError_TDCHisto   = 1;

    } else if ( ((*tdc_it).second).tdcError() & 0x0492 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " ROS " << code.getROS() << " ROB " << code.getROB()
	<< " TDC L1 buffer overflow in TDC " << (*tdc_it).first << endl;

      tdcError_ROSSummary = 16;
      tdcError_ROSError   = 13;
      tdcError_TDCHisto   = 2;

    } else if ( ((*tdc_it).second).tdcError() & 0x2000 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " ROS " << code.getROS() << " ROB " << code.getROB()
	<< " TDC L1A FIFO overflow in TDC " << (*tdc_it).first << endl;

      tdcError_ROSSummary = 17;
      tdcError_ROSError   = 14;
      tdcError_TDCHisto   = 3;

    } else if ( ((*tdc_it).second).tdcError() & 0x0924 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " ROS " << code.getROS() << " ROB " << code.getROB()
	<< " TDC hit error in TDC " << (*tdc_it).first << endl;

      tdcError_ROSSummary = 18;
      tdcError_ROSError   = 15;
      tdcError_TDCHisto   = 4;

    } else if ( ((*tdc_it).second).tdcError() & 0x1000 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " ROS " << code.getROS() << " ROB " << code.getROB()
	<< " TDC hit rejected in TDC " << (*tdc_it).first << endl;

      tdcError_ROSSummary = 19;
      tdcError_ROSError   = 16;
      tdcError_TDCHisto   = 5;

    } else {
      LogWarning("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< " TDC error code not known " << ((*tdc_it).second).tdcError() << endl;
    }

    ROSSummary->Fill(tdcError_ROSSummary,code.getROS());

    if(tdcError_ROSSummary <= 15) {
      eventErrorFlag = true;
    }

    if(mode <= 2) {
      ROSError->Fill(tdcError_ROSError,(*tdc_it).first);
      if(mode <= 1)
	rosHistos["TDCError"][code.getROSID()]->Fill(tdcError_TDCHisto+6*((*tdc_it).second).tdcID(),(*tdc_it).first);
    }
  }

  // Read SC data
  if (mode <= 1 && getSCInfo) {
    // NumberOf16bitWords counts the # of words + 1 subheader
    // the SC includes the SC "private header" and the ROS header and trailer (= NumberOf16bitWords +3)
    rosHistos["SCSizeVsROSSize"][code.getSCID()]->Fill(ros,data.getSCTrailer().wordCount());

  }
}

void DTDataIntegrityTask::processFED(DTDDUData & data, const std::vector<DTROS25Data> & rosData, int ddu) {

  neventsDDU++;
  if (neventsDDU%1000 == 0)
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << "[DTDataIntegrityTask]: " << neventsDDU << " events analyzed by processFED" << endl;


  DTROChainCoding code;
  code.setDDU(ddu);
  if(code.getDDUID() < FEDIDmin || code.getDDUID() > FEDIDMax) return;

  hFEDEntry->Fill(code.getDDUID());

  FEDTrailer trailer = data.getDDUTrailer();
  FEDHeader header = data.getDDUHeader();

  // check consistency of header and trailer
  if(!header.check()) {
    // error code 7
    hFEDFatal->Fill(code.getDDUID());
    hCorruptionSummary->Fill(code.getDDUID(), 7);
  }

  if(!trailer.check()) {
    // error code 8
    hFEDFatal->Fill(code.getDDUID());
    hCorruptionSummary->Fill(code.getDDUID(), 8);
  }

  // check CRC error bit set by DAQ before sending data on SLink
  if(data.crcErrorBit()) {
    // error code 6
    hFEDFatal->Fill(code.getDDUID());
    hCorruptionSummary->Fill(code.getDDUID(), 6);
  }

  DTDDUSecondStatusWord secondWord = data.getSecondStatusWord();

  // Fill the status summary of the TTS

  //1D HISTO WITH TTS VALUES form trailer (7 bins = 7 values)
  int ttsCodeValue = -1;
  int ttsSummaryBin = -1;

  switch(trailer.ttsBits()) {
  case 0:{ //disconnected
    ttsCodeValue = 0;
    break;
  }
  case 1:{ //warning overflow
    ttsCodeValue = 1;
    if(secondWord.warningROSPAF()) { // ROS PAF
      ttsSummaryBin = 1;
    } else { // DDU PAF
      ttsSummaryBin = 2;
    }

    break;
  }
  case 2:{ //out of sinch
    ttsCodeValue = 2;
    bool knownOrigin = false;
    if(secondWord.outOfSynchROSError()) {// ROS Error
      ttsSummaryBin = 7;
      knownOrigin = true;
    }
    if(secondWord.l1AIDError()) {// L1A Mism.
      ttsSummaryBin = 6;
      knownOrigin = true;
    }
    if(secondWord.bxIDError()) {// BX Mism.
      ttsSummaryBin = 8;
      knownOrigin = true;
    }
    if(secondWord.outputFifoFull() || secondWord.inputFifoFull() || secondWord.fifoFull()) { // DDU Full
      ttsSummaryBin = 5;
      knownOrigin = true;
    }
    if(!knownOrigin) ttsSummaryBin = 9; // Error in DDU logic

    break;
  }
  case 4:{ //busy
    ttsCodeValue = 3;
    bool knownOrigin = false;
    if(secondWord.busyROSPAF()) { // ROS PAF
      ttsSummaryBin = 3;
      knownOrigin = true;
    }
    if(secondWord.outputFifoAlmostFull() || secondWord.inputFifoAlmostFull() || secondWord.fifoAlmostFull() ){ // DDU PAF
      ttsSummaryBin = 4;
      knownOrigin = true;
    }
    if(!knownOrigin) ttsSummaryBin = 9; // Error in DDU logic
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
    ttsCodeValue = 7;
  }
  }
  if(mode <= 1) dduHistos["TTSValues"][code.getDDUID()]->Fill(ttsCodeValue);
  if(ttsSummaryBin != -1) {
    hTTSSummary->Fill(ddu, ttsSummaryBin);
  }






  //2D HISTO: ROS VS STATUS (8 BIT = 8 BIN) from 1st-2nd status words (9th BIN FROM LIST OF ROS in 2nd status word)
  MonitorElement* hROSStatus = dduHistos["ROSStatus"][code.getDDUID()];
  //1D HISTO: NUMBER OF ROS IN THE EVENTS from 2nd status word

  int rosList = secondWord.rosList();
  set<int> rosPositions;
  for(int i=0;i<12;i++) {
    if(rosList & 0x1) {
      rosPositions.insert(i);
      //9th BIN FROM LIST OF ROS in 2nd status word
      if(mode <= 2) hROSStatus->Fill(8,i,1);
    }
    rosList >>= 1;
  }

  int channel=0;
  for (vector<DTDDUFirstStatusWord>::const_iterator fsw_it = data.getFirstStatusWord().begin();
       fsw_it != data.getFirstStatusWord().end(); fsw_it++) {
    // assuming association one-to-one between DDU channel and ROS
    if(mode <= 2) {
      hROSStatus->Fill(0,channel,(*fsw_it).channelEnabled());
      hROSStatus->Fill(1,channel,(*fsw_it).timeout());
      hROSStatus->Fill(2,channel,(*fsw_it).eventTrailerLost());
      hROSStatus->Fill(3,channel,(*fsw_it).opticalFiberSignalLost());
      hROSStatus->Fill(4,channel,(*fsw_it).tlkPropagationError());
      hROSStatus->Fill(5,channel,(*fsw_it).tlkPatternError());
      hROSStatus->Fill(6,channel,(*fsw_it).tlkSignalLost());
      hROSStatus->Fill(7,channel,(*fsw_it).errorFromROS());
    }
    // check that the enabled channel was also in the read-out
    if((*fsw_it).channelEnabled() == 1 &&
       rosPositions.find(channel) == rosPositions.end()) {
      if(mode <= 2) hROSStatus->Fill(9,channel,1);
      // error code 1
      hFEDFatal->Fill(code.getDDUID());
      hCorruptionSummary->Fill(code.getDDUID(), 1);
    }
    channel++;
  }


  // ---------------------------------------------------------------------
  // cross checks between FED and ROS data
  // check the BX ID against the ROSs
  set<int> rosBXIds = rosBxIdsPerFED[ddu];
  if((rosBXIds.size() > 1 || rosBXIds.find(header.bxID()) == rosBXIds.end()) && rosBXIds.size() != 0) { // in this case look for faulty ROSs
    for(vector<DTROS25Data>::const_iterator rosControlData = rosData.begin();
	rosControlData != rosData.end(); ++rosControlData) { // loop over the ROS data
      for (vector<DTROSDebugWord>::const_iterator debug_it = (*rosControlData).getROSDebugs().begin();
	   debug_it != (*rosControlData).getROSDebugs().end(); debug_it++) { // Loop over ROS debug words
	if ((*debug_it).debugType() == 0 && (*debug_it).debugMessage() != header.bxID()) { // check the BX
	  int ros = (*rosControlData).getROSID();
	  // fill the error bin
	  if(mode <= 2) hROSStatus->Fill(11,ros-1);
	  // error code 2
	  hFEDFatal->Fill(code.getDDUID());
	  hCorruptionSummary->Fill(code.getDDUID(), 2);
	}
      }
    }
  }

  // check the BX ID against other FEDs
  fedBXIds.insert(header.bxID());
  if(fedBXIds.size() != 1) {
    LogWarning("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << "ERROR: FED " << ddu << " BX ID different from other feds: " << header.bxID() << endl;
    // error code 3
    hFEDFatal->Fill(code.getDDUID());
    hCorruptionSummary->Fill(code.getDDUID(), 3);
  }


  // check the L1A ID against the ROSs
  set<int> rosL1AIds = rosL1AIdsPerFED[ddu];
  if((rosL1AIds.size() > 1 || rosL1AIds.find(header.lvl1ID()-1) == rosL1AIds.end()) && rosL1AIds.size() != 0) { // in this case look for faulty ROSs
    //If L1A_ID error identify which ROS has wrong L1A
    for (vector<DTROS25Data>::const_iterator rosControlData = rosData.begin();
	 rosControlData != rosData.end(); rosControlData++) { // loop over the ROS data
      int ROSHeader_TTCCount = ((*rosControlData).getROSHeader().TTCEventCounter() + 1) % 0x1000000; // fix comparison in case of last counting bin in ROS /first one in DDU
      if( ROSHeader_TTCCount != header.lvl1ID() ) {
	int ros = (*rosControlData).getROSID();
	if(mode <= 2) hROSStatus->Fill(10,ros-1);
	// error code 4
	hFEDFatal->Fill(code.getDDUID());
	hCorruptionSummary->Fill(code.getDDUID(), 4);
      }
    }
  }

  //1D HISTOS: EVENT LENGHT from trailer
  int fedEvtLenght = trailer.lenght()*8;
  //   if(fedEvtLenght > 16000) fedEvtLenght = 16000; // overflow bin
  dduHistos["EventLenght"][code.getDDUID()]->Fill(fedEvtLenght);

  if(mode > 1) return;

  dduTimeHistos["FEDAvgEvLenghtvsLumi"][code.getDDUID()]->accumulateValueTimeSlot(fedEvtLenght);

  // size of the list of ROS in the Read-Out
  dduHistos["ROSList"][code.getDDUID()]->Fill(rosPositions.size());


  //2D HISTO: FIFO STATUS from 2nd status word
  MonitorElement *hFIFOStatus = dduHistos["FIFOStatus"][code.getDDUID()];
  int inputFifoFull = secondWord.inputFifoFull();
  int inputFifoAlmostFull = secondWord.inputFifoAlmostFull();
  int fifoFull = secondWord.fifoFull();
  int fifoAlmostFull = secondWord.fifoAlmostFull();
  int outputFifoFull = secondWord.outputFifoFull();
  int outputFifoAlmostFull = secondWord.outputFifoAlmostFull();
  for(int i=0;i<3;i++){
    if(inputFifoFull & 0x1){
      hFIFOStatus->Fill(i,0);
    }
    if(inputFifoAlmostFull & 0x1){
      hFIFOStatus->Fill(i,1);
    }
    if(fifoFull & 0x1){
      hFIFOStatus->Fill(3+i,0);
    }
    if(fifoAlmostFull & 0x1){
      hFIFOStatus->Fill(3+i,1);
    }
    if(!(inputFifoFull & 0x1) && !(inputFifoAlmostFull & 0x1)){
      hFIFOStatus->Fill(i,2);
    }
    if(!(fifoFull & 0x1) && !(fifoAlmostFull & 0x1)){
      hFIFOStatus->Fill(3+i,2);
    }
    inputFifoFull >>= 1;
    inputFifoAlmostFull >>= 1;
    fifoFull >>= 1;
    fifoAlmostFull >>= 1;
  }

  if(outputFifoFull){
    hFIFOStatus->Fill(6,0);
  }
  if(outputFifoAlmostFull){
    hFIFOStatus->Fill(6,1);
  }
  if(!outputFifoFull && !outputFifoAlmostFull){
    hFIFOStatus->Fill(6,2);
  }







  //1D HISTO: EVENT TYPE from header
  dduHistos["EventType"][code.getDDUID()]->Fill(header.triggerType());

  // fill the distribution of the BX ids
  dduHistos["BXID"][code.getDDUID()]->Fill(header.bxID());

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

std::string DTDataIntegrityTask::topFolder(bool isFEDIntegrity) const {

  string folder = isFEDIntegrity ? fedIntegrityFolder : "DT/00-DataIntegrity";

  if (!isFEDIntegrity)
    folder += (mode==1) ? "_SM/" : (mode==3) ? "_EvF/" : "/";

  return folder;

}

void DTDataIntegrityTask::channelsInCEROS(int cerosId, int chMask, vector<int>& channels ){
  for (int iCh=0; iCh<6;++iCh) {
    if ((chMask >> iCh) & 0x1){
      channels.push_back(cerosId*6+iCh);
    }
  }
  return;
}

void DTDataIntegrityTask::channelsInROS(int cerosMask, vector<int>& channels){
  for (int iCeros=0; iCeros<5;++iCeros) {
    if ((cerosMask >> iCeros) & 0x1){
      for (int iCh=0; iCh<6;++iCh) {
	channels.push_back(iCeros*6+iCh);
      }
    }
  }
  return;
}

void DTDataIntegrityTask::beginLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) {

  nEventsLS = 0;

}

void DTDataIntegrityTask::endLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) {

  int lumiBlock = ls.luminosityBlock();

  map<std::string, map<int, DTTimeEvolutionHisto*> >::iterator dduIt  = dduTimeHistos.begin();
  map<std::string, map<int, DTTimeEvolutionHisto*> >::iterator dduEnd = dduTimeHistos.end();
  for(; dduIt!=dduEnd; ++dduIt) {
    map<int, DTTimeEvolutionHisto*>::iterator histoIt  = dduIt->second.begin();
    map<int, DTTimeEvolutionHisto*>::iterator histoEnd = dduIt->second.end();
    for(; histoIt!=histoEnd; ++histoIt) {
      histoIt->second->updateTimeSlot(lumiBlock,nEventsLS);
    }
  }

  map<std::string, map<int, DTTimeEvolutionHisto*> >::iterator rosIt  = rosTimeHistos.begin();
  map<std::string, map<int, DTTimeEvolutionHisto*> >::iterator rosEnd = rosTimeHistos.end();
  for(; rosIt!=rosEnd; ++rosIt) {
    map<int, DTTimeEvolutionHisto*>::iterator histoIt  = rosIt->second.begin();
    map<int, DTTimeEvolutionHisto*>::iterator histoEnd = rosIt->second.end();
    for(; histoIt!=histoEnd; ++histoIt) {
      histoIt->second->updateTimeSlot(lumiBlock,nEventsLS);
    }
  }

}

void DTDataIntegrityTask::analyze(const edm::Event& e, const edm::EventSetup& c)
{
  nevents++;
  nEventMonitor->Fill(nevents);

  nEventsLS++;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: preProcessEvent" <<endl;
  // clear the set of BXids from the ROSs
  for(map<int, set<int> >::iterator rosBxIds = rosBxIdsPerFED.begin(); rosBxIds != rosBxIdsPerFED.end(); ++rosBxIds) {
    (*rosBxIds).second.clear();
  }

  fedBXIds.clear();

  for(map<int, set<int> >::iterator rosL1AIds = rosL1AIdsPerFED.begin(); rosL1AIds != rosL1AIdsPerFED.end(); ++rosL1AIds) {
    (*rosL1AIds).second.clear();
  }

  // reset the error flag
  eventErrorFlag = false;

  // Digi collection
  edm::Handle<DTDDUCollection> dduProduct;
  e.getByToken(dduToken, dduProduct);
  edm::Handle<DTROS25Collection> ros25Product;
  e.getByToken(ros25Token, ros25Product);

  DTDDUData dduData;
  std::vector<DTROS25Data> ros25Data;

  if(dduProduct.isValid() && ros25Product.isValid()) {
    for(unsigned int i=0; i<dduProduct->size(); ++i)
    {
      dduData = dduProduct->at(i);
      ros25Data = ros25Product->at(i);
      // FIXME: passing id variable is not needed anymore - change processFED interface for next release!
      FEDHeader header = dduData.getDDUHeader();
      int id = header.sourceID();
      processFED(dduData, ros25Data, id);
      for(unsigned int j=0; j < ros25Data.size(); ++j) {
        int rosid = j+1;
        processROS25(ros25Data[j],id,rosid);
      }
    }
  }
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
