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

#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace edm;

DTDataIntegrityTask::DTDataIntegrityTask(const edm::ParameterSet& ps) : nevents(0) {

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
  << "[DTDataIntegrityTask]: Constructor" <<endl;

  checkUros = ps.getUntrackedParameter<bool>("checkUros",true);

  if (checkUros) {
	fedToken = consumes<DTuROSFEDDataCollection>(ps.getParameter<InputTag>("dtFEDlabel")); 
	FEDIDmin = FEDNumbering::MINDTUROSFEDID;
	FEDIDmax = FEDNumbering::MAXDTUROSFEDID;
	}
  else { dduToken = consumes<DTDDUCollection>(ps.getParameter<InputTag>("dtDDULabel"));
         ros25Token = consumes<DTROS25Collection>(ps.getParameter<InputTag>("dtROS25Label")); 
  	 FEDIDmin = FEDNumbering::MINDTFEDID;
  	 FEDIDmax = FEDNumbering::MAXDTFEDID;
	}

  neventsFED = 0;
  neventsuROS = 0;


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
    <<"[DTDataIntegrityTask]: Destructor. Analyzed "<< neventsFED <<" events"<<endl;
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << "[DTDataIntegrityTask]: postEndJob called!" <<endl;
}

/*
  Folder Structure (ROS Legacy):
  - One folder for each DDU, named FEDn
  - Inside each DDU folder the DDU histos and the ROSn folder
  - Inside each ROS folder the ROS histos and the ROBn folder
  - Inside each ROB folder one occupancy plot and the TimeBoxes
  with the chosen granularity (simply change the histo name)
 
  uROS (starting 2018):
  - 3 uROS Summary plots: Wheel-1/-2 (FED1369), Wheel0 (FED1370), Wheel+1/+2 (FED1371)
  - One folder for each FED
  - Inside each FED folder the uROSStatus histos, FED histos
  - One folder for each wheel and the corresponding ROSn folders
  - Inside each ROS folder the TDC and ROS errors histos, 24 Links/plot
*/

void DTDataIntegrityTask::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const & iRun, edm::EventSetup const & iSetup) {

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: postBeginJob" <<endl;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask] Get DQMStore service" << endl;

  // Loop over the DT FEDs

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << " FEDS: " << FEDIDmin  << " to " <<  FEDIDmax << " in the RO" << endl;

  // book FED integrity histos
  bookHistos(ibooker, FEDIDmin, FEDIDmax);

  if (checkUros){ //uROS starting on 2018
  // static booking of the histograms

  if(mode == 0 || mode ==2){
  for(int fed = FEDIDmin; fed <= FEDIDmax; ++fed) { // loop over the FEDs in the readout

    bookHistos(ibooker, string("FED"), fed);

    bookHistos(ibooker, string("CRATE"), fed);

    for(int uRos = 1; uRos <= NuROS; ++uRos) {// loop over all ROS
      bookHistosuROS(ibooker,fed,uRos);
    }
   }

    for (int wheel=-2;wheel<3;++wheel){
	for(int ros = 1; ros <= NuROS; ++ros) {// loop over all ROS
      bookHistosROS(ibooker,wheel,ros);
    }
   }

  } //Not in HLT or SM mode
  } //uROS
 
  else{ //Legacy ROS

  if(mode == 0 || mode ==2){
  // static booking of the histograms

  for(int fed = FEDIDmin; fed <= FEDIDmax; ++fed) { // loop over the FEDs in the readout
    DTROChainCoding code;
    code.setDDU(fed);
    bookHistos(ibooker, string("ROS_S"), code);

    bookHistos(ibooker, string("DDU"), code);

    for(int ros = 1; ros <= NuROS; ++ros) {// loop over all ROS
      code.setROS(ros);
      bookHistosROS25(ibooker, code);
    }
   }
  } //Not in HLT or SM mode
  } //Legacy ROS
}

void DTDataIntegrityTask::bookHistos(DQMStore::IBooker & ibooker, const int fedMin, const int fedMax) {

  ibooker.setCurrentFolder("DT/EventInfo/Counters");
  nEventMonitor = ibooker.bookFloat("nProcessedEventsDataIntegrity");

  // Standard FED integrity histos
  ibooker.setCurrentFolder(topFolder(true));

  int nFED = (fedMax - fedMin)+1;

  hFEDEntry = ibooker.book1D("FEDEntries","# entries per DT FED",nFED,fedMin,fedMax+1);
  
  if(checkUros){

  if(mode == 3 || mode ==1) {
	//Booked for completion in general CMS FED test. Not filled
        hFEDFatal = ibooker.book1D("FEDFatal","# fatal errors DT FED",nFED,fedMin,fedMax+1); //No available in uROS
        hFEDNonFatal = ibooker.book1D("FEDNonFatal","# NON fatal errors DT FED",nFED,fedMin,fedMax+1); //No available in uROS
        return; //Avoid duplication of Info in FEDIntegrity_EvF
        }

  string histoType = "ROSSummary";
  for (int wheel=-2;wheel<3;++wheel){
    string wheel_s = to_string(wheel);
    string histoName = "ROSSummary_W"+wheel_s;
    string fed_s = to_string(FEDIDmin+1); //3 FEDs from 2018 onwards
    if(wheel<0) fed_s = to_string(FEDIDmin);
    else if(wheel>0) fed_s = to_string(FEDIDmax);
    string histoTitle = "Summary Wheel" + wheel_s + " (FED " + fed_s + ")";

    ((summaryHistos[histoType])[wheel]) = ibooker.book2D(histoName,histoTitle,11,0,11,12,1,13);
    MonitorElement *histo = ((summaryHistos[histoType])[wheel]);
    histo ->setBinLabel(1,"Error 1",1);
    histo ->setBinLabel(2,"Error 2",1);
    histo ->setBinLabel(3,"Error 3",1);
    histo ->setBinLabel(4,"Error 4",1);
    histo ->setBinLabel(5,"Not OKflag",1);
    // TDC error bins
    histo ->setBinLabel(6,"TDC Fatal",1);
    histo ->setBinLabel(7,"TDC RO FIFO ov.",1);
    histo ->setBinLabel(8,"TDC L1 buf. ov.",1);
    histo ->setBinLabel(9,"TDC L1A FIFO ov.",1);
    histo ->setBinLabel(10,"TDC hit err.",1);
    histo ->setBinLabel(11,"TDC hit rej.",1);

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
  }

  else{ //if(!checkUros){
  hFEDFatal = ibooker.book1D("FEDFatal","# fatal errors DT FED",nFED,fedMin,fedMax+1);
  hFEDNonFatal = ibooker.book1D("FEDNonFatal","# NON fatal errors DT FED",nFED,fedMin,fedMax+1);

  if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

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
  hCorruptionSummary->setBinLabel(8,"Trailer Check",2);
  } //!checkUros
}

// ******************uROS******************** //
void DTDataIntegrityTask::bookHistos(DQMStore::IBooker & ibooker, string folder, const int fed){

  string wheel = "ZERO";
  if (fed ==FEDIDmin) wheel="NEG";
  else if(fed==FEDIDmax) wheel="POS";
  string fed_s = to_string(fed);
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << " Booking histos for FED: " << fed_s
    << " folder: " << folder << endl;

  string histoType;
  string histoName;
  string histoTitle;
  MonitorElement* histo = nullptr;

  // Crate (old DDU) Histograms
  if (folder == "CRATE") {

    ibooker.setCurrentFolder(topFolder(false) + "FED" + fed_s);

    histoType = "EventLength";
    histoName = "FED" + fed_s + "_" + histoType;
    histoTitle = "Event Length (Bytes) FED " +  fed_s;
    (fedHistos[histoType])[fed] = ibooker.book1D(histoName,histoTitle,501,0,30000);

    if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF
 
    histoType = "uROSStatus";
    histoName = "FED" + fed_s + "_" + histoType;
    (fedHistos[histoType])[fed] = ibooker.book2D(histoName,histoName,12,0,12,12,1,13);
    histo = (fedHistos[histoType])[fed];
    // only placeholders for the moment
    histo->setBinLabel(1,"Error G 1",1);
    histo->setBinLabel(2,"Error G 2",1);
    histo->setBinLabel(3,"Error G 3",1);
    histo->setBinLabel(4,"Error G 4",1);
    histo->setBinLabel(5,"Error G 5",1);
    histo->setBinLabel(6,"Error G 6",1);
    histo->setBinLabel(7,"Error G 7",1);
    histo->setBinLabel(8,"Error G 8",1);
    histo->setBinLabel(9,"Error G 9",1);
    histo->setBinLabel(10,"Error G 10",1);
    histo->setBinLabel(11,"Error G 11",1);
    histo->setBinLabel(12,"Error G 12",1);

    histo->setBinLabel(1,"uROS 1",2);
    histo->setBinLabel(2,"uROS 2",2);
    histo->setBinLabel(3,"uROS 3",2);
    histo->setBinLabel(4,"uROS 4",2);
    histo->setBinLabel(5,"uROS 5",2);
    histo->setBinLabel(6,"uROS 6",2);
    histo->setBinLabel(7,"uROS 7",2);
    histo->setBinLabel(8,"uROS 8",2);
    histo->setBinLabel(9,"uROS 9",2);
    histo->setBinLabel(10,"uROS 10",2);
    histo->setBinLabel(11,"uROS 11",2);
    histo->setBinLabel(12,"uROS 12",2);
    
    if(mode > 0) return; //Info for Online only

    histoType = "FEDAvgEvLengthvsLumi";
    histoName = "FED" + fed_s + "_" + histoType;
    histoTitle = "Avg Event Length (Bytes) vs LumiSec FED " +  fed_s;
    (fedTimeHistos[histoType])[fed] = new DTTimeEvolutionHisto(ibooker,histoName,histoTitle,200,10,true,0);
    
    histoType = "TTSValues";
    histoName = "FED" + fed_s + "_" + histoType;
    (fedHistos[histoType])[fed] = ibooker.book1D(histoName,histoName,8,0,8);
    histo = (fedHistos[histoType])[fed];
    histo->setBinLabel(1,"Disconnected",1);
    histo->setBinLabel(2,"Overflow Warning ",1);
    histo->setBinLabel(3,"Out of synch",1);
    histo->setBinLabel(4,"Busy",1);
    histo->setBinLabel(5,"Ready",1);
    histo->setBinLabel(6,"Error",1);
    histo->setBinLabel(7,"Disconnected",1);
    histo->setBinLabel(8,"Unknown",1);

    histoType = "uROSList";
    histoName = "FED" + fed_s + "_" + histoType;
    histoTitle = "# of uROS in the FED payload (FED" + fed_s + ")";
    (fedHistos[histoType])[fed] = ibooker.book1D(histoName,histoTitle,13,0,13);

    histoType = "BXID";
    histoName = "FED" + fed_s + "_BXID";
    histoTitle = "Distrib. BX ID (FED" + fed_s + ")";
    (fedHistos[histoType])[fed] = ibooker.book1D(histoName,histoTitle,3600,0,3600);
    
  }

  // uROS Histograms
  if ( folder == "FED" ) { // The summary of the error of the ROS on the same FED
    ibooker.setCurrentFolder(topFolder(false));

    if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

    histoType = "uROSSummary";
    histoName = "FED" + fed_s + "_uROSSummary";
    string histoTitle = "Summary Wheel" + wheel + " (FED " + fed_s + ")";

    ((summaryHistos[histoType])[fed]) = ibooker.book2D(histoName,histoTitle,12,0,12,12,1,13);
    MonitorElement *histo = ((summaryHistos[histoType])[fed]);
    // ROS error bins
    // Placeholders for Global Errors for the moment
    histo ->setBinLabel(1,"Error G 1",1);
    histo ->setBinLabel(2,"Error G 2",1);
    histo ->setBinLabel(3,"Error G 3",1);
    histo ->setBinLabel(4,"Error G 4",1);
    histo ->setBinLabel(5,"Error G 5",1);
    histo ->setBinLabel(6,"Error G 6",1);
    histo ->setBinLabel(7,"Error G 7",1);
    histo ->setBinLabel(8,"Error G 8",1);
    histo ->setBinLabel(9,"Error G 9",1);
    histo ->setBinLabel(10,"Error G 10",1);
    histo ->setBinLabel(11,"Error G 11",1);
    histo ->setBinLabel(12,"Error G 12",1);

    histo ->setBinLabel(1,"uROS1",2);
    histo ->setBinLabel(2,"uROS2",2);
    histo ->setBinLabel(3,"uROS3",2);
    histo ->setBinLabel(4,"uROS4",2);
    histo ->setBinLabel(5,"uROS5",2);
    histo ->setBinLabel(6,"uROS6",2);
    histo ->setBinLabel(7,"uROS7",2);
    histo ->setBinLabel(8,"uROS8",2);
    histo ->setBinLabel(9,"uROS9",2);
    histo ->setBinLabel(10,"uROS10",2);
    histo ->setBinLabel(11,"uROS11",2);
    histo ->setBinLabel(12,"uROS12",2);


    
  }

}
// ******************End uROS******************** //


void DTDataIntegrityTask::bookHistos(DQMStore::IBooker & ibooker, string folder, DTROChainCoding code) {

  string dduID_s = to_string(code.getDDU());
  string rosID_s = to_string(code.getROS());
  string robID_s = to_string(code.getROB());
  int wheel = (code.getDDUID() - 770)%5 - 2;
  string wheel_s = to_string(wheel);

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
    << " Booking histos for FED: " << code.getDDU() << " ROS: " << code.getROS()
    << " ROB: " << code.getROB() << " folder: " << folder << endl;

  string histoType;
  string histoName;
  string histoTitle;
  MonitorElement* histo = nullptr;

  // DDU Histograms
  if (folder == "DDU") {

    ibooker.setCurrentFolder(topFolder(false) + "FED" + dduID_s);

    histoType = "EventLength";
    histoName = "FED" + dduID_s + "_" + histoType;
    histoTitle = "Event Length (Bytes) FED " +  dduID_s;
    (fedHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoTitle,501,0,16032);

    if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

    histoType = "ROSStatus";
    histoName = "FED" + dduID_s + "_" + histoType;
    (fedHistos[histoType])[code.getDDUID()] = ibooker.book2D(histoName,histoName,12,0,12,12,0,12);
    histo = (fedHistos[histoType])[code.getDDUID()];
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

    if(mode > 0) return;  //Info for Online only

    histoType = "FEDAvgEvLengthvsLumi";
    histoName = "FED" + dduID_s + "_" + histoType;
    histoTitle = "Avg Event Length (Bytes) vs LumiSec FED " +  dduID_s;
    fedTimeHistos[histoType][code.getDDUID()] = new DTTimeEvolutionHisto(ibooker,histoName,histoTitle,200,10,true,0);

    histoType = "TTSValues";
    histoName = "FED" + dduID_s + "_" + histoType;
    (fedHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoName,8,0,8);
    histo = (fedHistos[histoType])[code.getDDUID()];
    histo->setBinLabel(1,"disconnected",1);
    histo->setBinLabel(2,"warning overflow",1);
    histo->setBinLabel(3,"out of synch",1);
    histo->setBinLabel(4,"busy",1);
    histo->setBinLabel(5,"ready",1);
    histo->setBinLabel(6,"error",1);
    histo->setBinLabel(7,"disconnected",1);
    histo->setBinLabel(8,"unknown",1);

    histoType = "EventType";
    histoName = "FED" + dduID_s + "_" + histoType;
    (fedHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoName,2,1,3);
    histo = (fedHistos[histoType])[code.getDDUID()];
    histo->setBinLabel(1,"physics",1);
    histo->setBinLabel(2,"calibration",1);

    histoType = "ROSList";
    histoName = "FED" + dduID_s + "_" + histoType;
    histoTitle = "# of ROS in the FED payload (FED" + dduID_s + ")";
    (fedHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoTitle,13,0,13);

    histoType = "FIFOStatus";
    histoName = "FED" + dduID_s + "_" + histoType;
    (fedHistos[histoType])[code.getDDUID()] = ibooker.book2D(histoName,histoName,7,0,7,3,0,3);
    histo = (fedHistos[histoType])[code.getDDUID()];
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
    histoName = "FED" + dduID_s + "_BXID";
    histoTitle = "Distrib. BX ID (FED" + dduID_s + ")";
    (fedHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName,histoTitle,3600,0,3600);

  }

  // ROS Histograms
  if ( folder == "ROS_S" ) { // The summary of the error of the ROS on the same FED
    ibooker.setCurrentFolder(topFolder(false));

    if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

    histoType = "ROSSummary";
    histoName = "FED" + dduID_s + "_ROSSummary";
    string histoTitle = "Summary Wheel" + wheel_s + " (FED " + dduID_s + ")";

    ((summaryHistos[histoType])[code.getDDUID()]) = ibooker.book2D(histoName,histoTitle,20,0,20,12,1,13);
    MonitorElement *histo = ((summaryHistos[histoType])[code.getDDUID()]);
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
    ibooker.setCurrentFolder(topFolder(false) + "FED" + dduID_s + "/" + folder + rosID_s);

    if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

    histoType = "ROSError";
    histoName = "FED" + dduID_s + "_" + folder + rosID_s + "_ROSError";
    histoTitle = histoName + " (ROBID error summary)";
    if(mode < 1) //Online only
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
    if(mode < 1) { //Online only
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

    histoType = "ROSEventLength";
    histoName = "FED" + dduID_s + "_" + folder + rosID_s + "_ROSEventLength";
    histoTitle = "Event Length (Bytes) FED " +  dduID_s + " ROS " + rosID_s;
    (rosHistos[histoType])[code.getROSID()] = ibooker.book1D(histoName,histoTitle,101,0,1616);

    histoType = "ROSAvgEventLengthvsLumi";
    histoName = "FED" + dduID_s + "_" + folder + rosID_s + histoType;
    histoTitle = "Event Length (Bytes) FED " +  dduID_s + " ROS " + rosID_s;
    rosTimeHistos[histoType][code.getROSID()] = new DTTimeEvolutionHisto(ibooker,histoName,histoTitle,200,10,true,0);

    histoType = "TDCError";
    histoName = "FED" + dduID_s + "_" + folder + rosID_s + "_TDCError";
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
    histoName = "FED" + dduID_s + "_" + "ROS" + rosID_s + "_ROB_mean";
    string fullName = topFolder(false) + "FED" + dduID_s + "/" + folder + rosID_s+ "/" + histoName;
    names.insert (pair<std::string,std::string> (histoType,string(fullName)));
    (rosHistos[histoType])[code.getROSID()] = ibooker.book2D(histoName,histoName,25,0,25,100,0,100);
    (rosHistos[histoType])[code.getROSID()]->setAxisTitle("ROB #",1);
    (rosHistos[histoType])[code.getROSID()]->setAxisTitle("ROB wordcounts",2);

  }

  // SC Histograms
  if ( folder == "SC" ) {
    // The plots are per wheel
    ibooker.setCurrentFolder(topFolder(false) + "FED" + dduID_s);
    if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

    // SC data Size
    histoType = "SCSizeVsROSSize";
    histoName = "FED" + dduID_s + "_SCSizeVsROSSize";
    histoTitle = "SC size vs SC (FED " + dduID_s + ")";
    rosHistos[histoType][code.getSCID()] = ibooker.book2D(histoName,histoTitle,12,1,13,51,-1,50);
    rosHistos[histoType][code.getSCID()]->setAxisTitle("SC",1);

  }
}


void DTDataIntegrityTask::bookHistosROS25(DQMStore::IBooker & ibooker, DTROChainCoding code) {
  
  bookHistos(ibooker, string("ROS"), code);

    if(mode < 1)
      if(getSCInfo)
	bookHistos(ibooker, string("SC"), code);
}

// ******************uROS******************** //
void DTDataIntegrityTask::bookHistosROS(DQMStore::IBooker & ibooker, const int wheel, const int ros){
  string wheel_s = to_string(wheel);
  string ros_s = to_string(ros);
  ibooker.setCurrentFolder(topFolder(false) + "Wheel" + wheel_s + "/ROS" + ros_s);

  	string histoType = "ROSError";
	int linkDown = 0; string linkDown_s = to_string(linkDown);
	int linkUp = linkDown+24;  string linkUp_s = to_string(linkUp);
  	string histoName = "W" + wheel_s + "_" + "ROS" + ros_s + "_"+histoType;
  	string histoTitle = histoName + " (Link " + linkDown_s +"-"+ linkUp_s + " error summary)";
	unsigned int keyHisto = (uROSError)*1000 + (wheel+2)*100 +(ros-1);
	if(mode < 1) // Online only
    		urosHistos[keyHisto] = ibooker.book2D(histoName,histoTitle,11,0,11,25,0,25);
  	else
    		urosHistos[keyHisto] = ibooker.book2D(histoName,histoTitle,5,0,5,25,0,25);

  	MonitorElement* histo = urosHistos[keyHisto];
  	// uROS error bins
  	// Placeholders for the moment
    	histo->setBinLabel(1,"Error 1",1);
    	histo->setBinLabel(2,"Error 2",1);
    	histo->setBinLabel(3,"Error 3",1);
    	histo->setBinLabel(4,"Error 4",1);
    	histo->setBinLabel(5,"Not OKFlag",1);
    	if(mode < 1) { //Online only
  	// TDC error bins
      	histo->setBinLabel(6,"TDC Fatal",1);
      	histo->setBinLabel(7,"TDC RO FIFO ov.",1);
      	histo->setBinLabel(8,"TDC L1 buf. ov.",1);
      	histo->setBinLabel(9,"TDC L1A FIFO ov.",1);
      	histo->setBinLabel(10,"TDC hit err.",1);
      	histo->setBinLabel(11,"TDC hit rej.",1);
    	}
   	for (int link=linkDown; link < (linkUp+1); ++link){
    		string link_s = to_string(link);
    		histo->setBinLabel(link+1,"Link"+link_s,2);
   	}
 	 


  if(mode > 1) return;

        histoType = "TDCError";
        linkDown = 0; linkDown_s = to_string(linkDown);
        linkUp = linkDown+24;  linkUp_s = to_string(linkUp);
        histoName = "W" + wheel_s + "_" + "ROS" + ros_s + "_"+histoType;
        histoTitle = histoName + " (Link " + linkDown_s +"-"+ linkUp_s + " error summary)";
 	keyHisto = (TDCError)*1000 + (wheel+2)*100 + (ros-1); 
    	urosHistos[keyHisto] = ibooker.book2D(histoName,histoTitle,24,0,24,25,0,25);
    	histo = urosHistos[keyHisto];
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

        for (int link=linkDown; link < (linkUp+1); ++link){
                string link_s = to_string(link);
                histo->setBinLabel(link+1,"Link"+link_s,2);
        }
} //bookHistosROS

void DTDataIntegrityTask::bookHistosuROS(DQMStore::IBooker & ibooker, const int fed, const int uRos){
  string fed_s = to_string(fed);
  string uRos_s = to_string(uRos);
  ibooker.setCurrentFolder(topFolder(false) + "FED" + fed_s + "/uROS" + uRos_s);

  if(mode > 1) return;

  string histoType = "uROSEventLength";
  string histoName = "FED" + fed_s + "_uROS" +  uRos_s + "_" + "EventLength";
  string histoTitle = "Event Length (Bytes) FED " +  fed_s + " uROS" + uRos_s;
  unsigned int keyHisto = (uROSEventLength)*1000 + (fed-FEDIDmin)*100 + (uRos-1);
  urosHistos[keyHisto] = ibooker.book1D(histoName,histoTitle,101,0,5000);

  histoType = "uROSAvgEventLengthvsLumi";
  histoName = "FED" + fed_s + "_ROS" +  uRos_s + "AvgEventLengthvsLumi";
  histoTitle = "Event Length (Bytes) FED " +  fed_s + " ROS" + uRos_s;
  keyHisto = (fed-FEDIDmin)*100 + (uRos-1);
  urosTimeHistos[keyHisto] = new DTTimeEvolutionHisto(ibooker,histoName,histoTitle,200,10,true,0);

  histoType = "TTSValues";
  histoName = "FED" + fed_s + "_" + "uROS" + uRos_s + "_" + histoType;
  keyHisto = TTSValues*1000 + (fed-FEDIDmin)*100 + (uRos-1);
  urosHistos[keyHisto] = ibooker.book1D(histoName,histoName,8,0,8);
  MonitorElement* histo = urosHistos[keyHisto];
  histo->setBinLabel(1,"Disconnected",1);
  histo->setBinLabel(2,"Overflow Warning ",1);
  histo->setBinLabel(3,"Out of synch",1);
  histo->setBinLabel(4,"Busy",1);
  histo->setBinLabel(5,"Ready",1);
  histo->setBinLabel(6,"Error",1);
  histo->setBinLabel(7,"Disconnected",1);
  histo->setBinLabel(8,"Unknown",1);

}
// ******************End uROS******************** //

// ******************uROS******************** //
void DTDataIntegrityTask::processuROS(DTuROSROSData & data, int fed, int uRos){

  neventsuROS++; // FIXME: implement a counter which makes sense

      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << "[DTDataIntegrityTask]: " << neventsuROS << " events analyzed by processuROS" << endl;

  if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

  MonitorElement* uROSSummary = nullptr;
  uROSSummary = summaryHistos["uROSSummary"][fed];

  MonitorElement* uROSStatus = nullptr;
  uROSStatus = fedHistos["uROSStatus"][fed];

  if(!uROSSummary) {
    LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") <<
        "Trying to access non existing ME at FED " << fed  <<
        std::endl;
    return;
  }

  unsigned int slotMap = ( data.getboardId() ) & 0xF;
  if (slotMap==0) return; //prevention for Simulation empty uROS data
  int ros = theROS(slotMap, 0); //first sector correspondign to link 0
  int ddu = theDDU(fed,  slotMap, 0, false);
  int wheel = (ddu - 770)%5 - 2;
  MonitorElement* ROSSummary = nullptr;
  ROSSummary = summaryHistos["ROSSummary"][wheel];

  // Summary of all Link errors
  MonitorElement* uROSError0 = nullptr;
  MonitorElement* uROSError1 = nullptr;
  MonitorElement* uROSError2 = nullptr;

  int errorX[5][12]={{0}}; //5th is notOK flag

  if(mode <= 2){

  if(uRos>2){ //sectors 1-12

    uROSError0 = urosHistos[(uROSError)*1000 + (wheel+2)*100 + (ros-1)]; //links 0-23
    uROSError1 = urosHistos[(uROSError)*1000 + (wheel+2)*100 + (ros)];   //links 24-47
    uROSError2 = urosHistos[(uROSError)*1000 + (wheel+2)*100 + (ros+1)]; //links 48-71

    if ((!uROSError2) || (!uROSError1) || (!uROSError0) ) {
      LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") <<
        "Trying to access non existing ME at uROS " << uRos  << 
        std::endl;
     return;
    }

  // uROS errors
   for (unsigned int link = 0; link<72; ++link){
      for (unsigned int flag = 0; flag<5; ++flag){
	if((data.getokxflag(link)>>flag) & 0x1) { // Undefined Flag 1-4 64bits word for each MTP (12 channels)
		int value = flag;
		if (flag==0) value=5; //move it to the 5th bin
			
		if (value>0){  
			       if(link<24) {
					errorX[value-1][ros-1]+=1;
					uROSError0->Fill(value-1,link); //bins start at 0 despite labelin
					}
                               else if(link<48) {
					errorX[value-1][ros]+=1;
					uROSError1->Fill(value-1,link-23);
					}
                               else if(link<72) {
					errorX[value-1][ros+1]+=1;
					uROSError2->Fill(value-1,link-47);
				}
		}//value>0	
	}//flag value
      } //loop on flags
    } //loop on links
  } //uROS>2

  else{//uRos<3
    
     for (unsigned int link = 0; link<12; ++link){
      for (unsigned int flag = 0; flag<5; ++flag){
        if((data.getokxflag(link)>>flag) & 0x1) {// Undefined Flag 1-4 64bits word for each MTP (12 channels)
		int value = flag;
		int sc = 24;
                if (flag==0) value=5; //move it to the 5th bin
                        
		if (value>0){
			unsigned int keyHisto = (uROSError)*1000 + (wheel+2)*100 +link; //ros -1 = link in this case
			uROSError0 = urosHistos[keyHisto];
			if(!uROSError0) {
		      		LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") <<
		        	"Trying to access non existing ME at uROS " << uRos  <<
		        	std::endl;
     				return;
			}	
			errorX[value-1][link]+=1; // ros-1=link in this case
			uROSError0->Fill(value-1,sc); //bins start at 0 despite labeling, this is the old SC
		}
	} //flag values
       } //loop on flags
      } //loop on links
  } //else uRos<3
		
  } //mode<=2

// Fill the ROSSummary (1 per wheel) histo
  for (unsigned int iros = 0; iros<12;++iros){
	for (unsigned int bin=0; bin<5;++bin){
    		if(errorX[bin][iros]!=0) ROSSummary->Fill(bin, iros+1); //bins start at 1 
	}
  }

// Global Errors for uROS
   for (unsigned int flag = 4; flag<16; ++flag){
      if((data.getuserWord()>>flag) & 0x1 ) {
		uROSSummary->Fill(flag-4, uRos);
		uROSStatus->Fill(flag-4, uRos); //duplicated info?
	}
   }

  // ROS error
  for (unsigned int icounter = 0; icounter<data.geterrors().size(); ++icounter){
    int link = data.geterrorROBID(icounter);
    int tdc = data.geterrorTDCID(icounter);
    int error = data.geterror(icounter);
    int tdcError_ROSSummary = 0;
    int tdcError_ROSError = 0;
    int tdcError_TDCHisto = 0;

    if(error & 0x4000 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << " ROS " << uRos << " ROB " << link
        << " Internal fatal Error 4000 in TDC " << error << endl;

      tdcError_ROSSummary = 5;
      tdcError_ROSError   = 5;
      tdcError_TDCHisto   = 0;

    } else if ( error & 0x0249 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << " ROS " << uRos << " ROB " << link
        << " TDC FIFO overflow in TDC " << error << endl;

      tdcError_ROSSummary = 6;
      tdcError_ROSError   = 6;
      tdcError_TDCHisto   = 1;

    } else if ( error & 0x0492 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << " ROS " << uRos << " ROB " << link
        << " TDC L1 buffer overflow in TDC " << error << endl;

      tdcError_ROSSummary = 7;
      tdcError_ROSError   = 7;
      tdcError_TDCHisto   = 2;

    } else if ( error & 0x2000 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << " ROS " << uRos << " ROB " << link
        << " TDC L1A FIFO overflow in TDC " << error << endl;

      tdcError_ROSSummary = 8;
      tdcError_ROSError   = 8;
      tdcError_TDCHisto   = 3;

    } else if ( error & 0x0924 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << " uROS " << uRos << " ROB " << link
        << " TDC hit error in TDC " << error << endl;

      tdcError_ROSSummary = 9;
      tdcError_ROSError   = 9;
      tdcError_TDCHisto   = 4;

    } else if ( error & 0x1000 ) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << " uROS " << uRos << " ROB " << link
        << " TDC hit rejected in TDC " << error << endl;

      tdcError_ROSSummary = 10;
      tdcError_ROSError   = 10;
      tdcError_TDCHisto   = 5;

    } else {
      LogWarning("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << " TDC error code not known " << error << endl;
    }

   if (uRos<3){
      ROSSummary->Fill(tdcError_ROSSummary,link+1); //link 0 = ROS 1
      int sc = 24;
      if(mode<=2) {urosHistos[(uROSError)*1000 + (wheel+2)*100 + (link)]->Fill(tdcError_ROSError,sc);
		 if(mode<=1) urosHistos[(TDCError)*1000 + (wheel+2)*100 + (link)]->Fill(tdcError_TDCHisto+6*(tdc-1),sc); // ros-1=link in this case
   	}//mode<=2
   } //uRos<3
   else{//uRos>2
      if (link<24) ROSSummary->Fill(tdcError_ROSSummary,ros); 
      else if (link<48) ROSSummary->Fill(tdcError_ROSSummary,ros+1);
      else if (link<72) ROSSummary->Fill(tdcError_ROSSummary,ros+2);

      if(mode<=2){
     	if (link<24)  	uROSError0->Fill(tdcError_ROSError,link);
     	else if (link<48)  uROSError1->Fill(tdcError_ROSError,link-23);
     	else if (link<72)  uROSError2->Fill(tdcError_ROSError,link-47);
     
     	if(mode<=1){
		if (link<24) urosHistos[(TDCError)*1000 + (wheel+2)*100 + (ros-1)]->Fill(tdcError_TDCHisto+6*(tdc-1),link);
		else if (link<48) urosHistos[(TDCError)*1000 + (wheel+2)*100 + (ros)]->Fill(tdcError_TDCHisto+6*(tdc-1),link-23);
		else if (link<72) urosHistos[(TDCError)*1000 + (wheel+2)*100 + (ros+1)]->Fill(tdcError_TDCHisto+6*(tdc-1),link-47);

     	} //mode<=1
      }	//mode<=2
   } //uROS>2
  } //loop on errors


  // 1D histograms for TTS values per uROS
  int ttsCodeValue = -1;

  int value = (data.getuserWord() & 0xF);
  switch(value) { 
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
  case 15:{ //disconnected
    ttsCodeValue = 6;
    break;
  }
  default:{
    LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      <<"[DTDataIntegrityTask] FED User control: wrong TTS value "<< value << " in FED " << fed << " uROS "<< uRos<< endl; //FIXME
    ttsCodeValue = 7;
  }
  }
  if(mode < 1) {
	urosHistos[TTSValues*1000 + (fed-FEDIDmin)*100 + (uRos-1)]->Fill(ttsCodeValue);

     // Plot the event length //NOHLT
        int uRosEventLength = (data.gettrailer() & 0xFFFFF)*8;
              urosTimeHistos[(fed-FEDIDmin)*100 + (uRos-1)]->accumulateValueTimeSlot(uRosEventLength);
     
              if(uRosEventLength > 5000) uRosEventLength = 5000;
              urosHistos[uROSEventLength*1000 + (fed-FEDIDmin)*100 + (uRos-1)]->Fill(uRosEventLength);
  }                      

}

// *****************End uROS ******************//

void DTDataIntegrityTask::processROS25(DTROS25Data & data, int ddu, int ros) {

  neventsuROS++; // FIXME: implement a counter which makes sense

      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
	<< "[DTDataIntegrityTask]: " << neventsuROS << " events analyzed by processROS25" << endl;

  // The ID of the RO board (used to map the histos)
  DTROChainCoding code;
  code.setDDU(ddu);
  code.setROS(ros);

  if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

  MonitorElement* ROSSummary = summaryHistos["ROSSummary"][code.getDDUID()];

  // Summary of all ROB errors
  MonitorElement* ROSError = nullptr;
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


  if(mode < 1) { // produce only when not in HLT or SM
    // ROB Trailer
    for (vector<DTROBTrailerWord>::const_iterator robt_it = data.getROBTrailers().begin();
	 robt_it != data.getROBTrailers().end(); robt_it++) { // loop over ROB trailers
      float  wCount = (*robt_it).wordCount()<100. ? (*robt_it).wordCount() : 99.9;
      rosHistos["ROB_mean"][code.getROSID()]->Fill((*robt_it).robID(),wCount);
    }

    // Plot the event length //NOHLT
    int rosEventLength = data.getROSTrailer().EventWordCount()*4;
    rosTimeHistos["ROSAvgEventLengthvsLumi"][code.getROSID()]->accumulateValueTimeSlot(rosEventLength);

    if(rosEventLength > 1600) rosEventLength = 1600;
    rosHistos["ROSEventLength"][code.getROSID()]->Fill(rosEventLength);
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
  if (mode < 1 && getSCInfo) {
    // NumberOf16bitWords counts the # of words + 1 subheader
    // the SC includes the SC "private header" and the ROS header and trailer (= NumberOf16bitWords +3)
    rosHistos["SCSizeVsROSSize"][code.getSCID()]->Fill(ros,data.getSCTrailer().wordCount());

  }
}

// ******************uROS******************** //
void DTDataIntegrityTask::processFED(DTuROSFEDData  & data, int fed){

  neventsFED++;
  if (neventsFED%1000 == 0)
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << "[DTDataIntegrityTask]: " << neventsFED << " events analyzed by processFED" << endl;

  if(fed < FEDIDmin || fed > FEDIDmax) return;

  hFEDEntry->Fill(fed);

  if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

  //1D HISTOS: EVENT LENGHT from trailer
  int fedEvtLength = data.getevtlgth()*8;  //1 word = 8 bytes
  //   if(fedEvtLength > 16000) fedEvtLength = 16000; // overflow bin
  fedHistos["EventLength"][fed]->Fill(fedEvtLength);

  if(mode > 1) return;

  fedTimeHistos["FEDAvgEvLengthvsLumi"][fed]->accumulateValueTimeSlot(fedEvtLength);

  // fill the distribution of the BX ids
  fedHistos["BXID"][fed]->Fill(data.getBXId()); 

 // size of the list of ROS in the Read-Out
  fedHistos["uROSList"][fed]->Fill(data.getnslots());

  // Fill the status summary of the TTS
  
  //1D HISTO WITH TTS VALUES form trailer (7 bins = 7 values)
     
  int ttsCodeValue = -1;
  int value = data.getTTS();
  switch(value) {
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
  case 15:{ //disconnected
    ttsCodeValue = 6;
    break;
  }
  default:{
    LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      <<"[DTDataIntegrityTask] FED TTS control: wrong TTS value "<< value << " in FED "<< fed<< endl;
    ttsCodeValue = 7;
  }
  }
  if(mode < 1) fedHistos["TTSValues"][fed]->Fill(ttsCodeValue);
 
}
// *****************End uROS ******************//


void DTDataIntegrityTask::processFED(DTDDUData & data, const std::vector<DTROS25Data> & rosData, int ddu) {

  neventsFED++;
  if (neventsFED%1000 == 0)
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << "[DTDataIntegrityTask]: " << neventsFED << " events analyzed by processFED" << endl;


  DTROChainCoding code;
  code.setDDU(ddu);
  if(code.getDDUID() < FEDIDmin || code.getDDUID() > FEDIDmax) return;

  hFEDEntry->Fill(code.getDDUID());

  const FEDTrailer& trailer = data.getDDUTrailer();
  const FEDHeader& header = data.getDDUHeader();

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

  if(mode == 3 || mode ==1) return; //Avoid duplication of Info in FEDIntegrity_EvF

  const DTDDUSecondStatusWord& secondWord = data.getSecondStatusWord();

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
  if(mode < 1) fedHistos["TTSValues"][code.getDDUID()]->Fill(ttsCodeValue);
  if(ttsSummaryBin != -1) {
    hTTSSummary->Fill(ddu, ttsSummaryBin);
  }


  //2D HISTO: ROS VS STATUS (8 BIT = 8 BIN) from 1st-2nd status words (9th BIN FROM LIST OF ROS in 2nd status word)
  MonitorElement* hROSStatus = fedHistos["ROSStatus"][code.getDDUID()];
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
  if((rosBXIds.size() > 1 || rosBXIds.find(header.bxID()) == rosBXIds.end()) && !rosBXIds.empty() ) { // in this case look for faulty ROSs
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
  if((rosL1AIds.size() > 1 || rosL1AIds.find(header.lvl1ID()-1) == rosL1AIds.end()) && !rosL1AIds.empty() ){ // in this case look for faulty ROSs
    //If L1A_ID error identify which ROS has wrong L1A
    for (vector<DTROS25Data>::const_iterator rosControlData = rosData.begin();
	 rosControlData != rosData.end(); rosControlData++) { // loop over the ROS data
      unsigned int ROSHeader_TTCCount = ((*rosControlData).getROSHeader().TTCEventCounter() + 1) % 0x1000000; // fix comparison in case of last counting bin in ROS /first one in DDU
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
  int fedEvtLength = trailer.fragmentLength()*8;
  //   if(fedEvtLength > 16000) fedEvtLength = 16000; // overflow bin
  fedHistos["EventLength"][code.getDDUID()]->Fill(fedEvtLength);

  if(mode > 1) return;

  fedTimeHistos["FEDAvgEvLengthvsLumi"][code.getDDUID()]->accumulateValueTimeSlot(fedEvtLength);

  // size of the list of ROS in the Read-Out
  fedHistos["ROSList"][code.getDDUID()]->Fill(rosPositions.size());


  //2D HISTO: FIFO STATUS from 2nd status word
  MonitorElement *hFIFOStatus = fedHistos["FIFOStatus"][code.getDDUID()];
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
  fedHistos["EventType"][code.getDDUID()]->Fill(header.triggerType());

  // fill the distribution of the BX ids
  fedHistos["BXID"][code.getDDUID()]->Fill(header.bxID());

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

  string folder = isFEDIntegrity ? fedIntegrityFolder : "DT/00-DataIntegrity/";

  if (mode == 0 || mode == 2)
    folder = "DT/00-DataIntegrity/"; //Move everything from FEDIntegrity except for SM and HLT modes

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

  if (checkUros){
  map<string, map<int, DTTimeEvolutionHisto*> >::iterator fedIt  = fedTimeHistos.begin();
  map<string, map<int, DTTimeEvolutionHisto*> >::iterator fedEnd = fedTimeHistos.end();
  for(; fedIt!=fedEnd; ++fedIt) {
    map<int, DTTimeEvolutionHisto*>::iterator histoIt  = fedIt->second.begin();
    map<int, DTTimeEvolutionHisto*>::iterator histoEnd = fedIt->second.end();
    for(; histoIt!=histoEnd; ++histoIt) {
      histoIt->second->updateTimeSlot(lumiBlock,nEventsLS);
    }
  }

  map<unsigned int, DTTimeEvolutionHisto*>::iterator urosIt  = urosTimeHistos.begin();
  map<unsigned int, DTTimeEvolutionHisto*>::iterator urosEnd = urosTimeHistos.end();
  for(; urosIt!=urosEnd; ++urosIt) {
        urosIt->second->updateTimeSlot(lumiBlock,nEventsLS);
  }

  }//uROS starting on 2018
  else{
  map<string, map<int, DTTimeEvolutionHisto*> >::iterator dduIt  = fedTimeHistos.begin();
  map<string, map<int, DTTimeEvolutionHisto*> >::iterator dduEnd = fedTimeHistos.end();
  for(; dduIt!=dduEnd; ++dduIt) {
    map<int, DTTimeEvolutionHisto*>::iterator histoIt  = dduIt->second.begin();
    map<int, DTTimeEvolutionHisto*>::iterator histoEnd = dduIt->second.end();
    for(; histoIt!=histoEnd; ++histoIt) {
      histoIt->second->updateTimeSlot(lumiBlock,nEventsLS);
    }
  }

  map<string, map<int, DTTimeEvolutionHisto*> >::iterator rosIt  = rosTimeHistos.begin();
  map<string, map<int, DTTimeEvolutionHisto*> >::iterator rosEnd = rosTimeHistos.end();
  for(; rosIt!=rosEnd; ++rosIt) {
    map<int, DTTimeEvolutionHisto*>::iterator histoIt  = rosIt->second.begin();
    map<int, DTTimeEvolutionHisto*>::iterator histoEnd = rosIt->second.end();
    for(; histoIt!=histoEnd; ++histoIt) {
      histoIt->second->updateTimeSlot(lumiBlock,nEventsLS);
    }
  }
  }//ROS Legacy
}

void DTDataIntegrityTask::analyze(const edm::Event& e, const edm::EventSetup& c)
{
  nevents++;
  nEventMonitor->Fill(nevents);

  nEventsLS++;

  LogTrace("DTRawToDigi|TDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: preProcessEvent" <<endl;

  if (checkUros){ //uROS starting on 2018
  // Digi collection
  edm::Handle<DTuROSFEDDataCollection> fedCol;
  e.getByToken(fedToken, fedCol);
  DTuROSFEDData fedData;
  DTuROSROSData urosData;

  if(fedCol.isValid()){
    for(unsigned int j=0; j <fedCol->size();++j){
    fedData = fedCol->at(j);
    int fed = fedData.getfed(); //argument should be void
    if (fed>FEDIDmax || fed<FEDIDmin) {
	LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: analyze, FED ID " << fed <<" not expected."  << endl;
	continue;
	}
    processFED(fedData, fed);

    if(mode == 3 || mode ==1) continue; //Not needed for FEDIntegrity_EvF 

    for(int slot=1; slot<=DOCESLOTS; ++slot)
    {
        urosData = fedData.getuROS(slot);
	if(fedData.getslotsize(slot)==0 || urosData.getslot()==-1) continue;
        processuROS(urosData,fed,slot);
      }
    }
  }
  } // checkUros
  else{ //Legacy ROS
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
      if (id>FEDIDmax || id<FEDIDmin) continue; //SIM uses extra FEDs not monitored

      processFED(dduData, ros25Data, id);
      for(unsigned int j=0; j < ros25Data.size(); ++j) {
        int rosid = j+1;
        processROS25(ros25Data[j],id,rosid);
      }
    }
  }
  }

}

// Conversions
int DTDataIntegrityTask::theDDU(int crate, int slot, int link, bool tenDDU) {

  int ros = theROS(slot,link);

  int ddu = 772;
  //if (crate == 1368) { ddu = 775; }
  //Needed just in case this FED should be used due to fibers length

  if (crate == FEDNumbering::MINDTUROSFEDID) {
    if (slot < 7)
      ddu = 770;
    else
      ddu = 771;
  }

  if (crate == (FEDNumbering::MINDTUROSFEDID+1)) { ddu = 772; }

  if (crate == FEDNumbering::MAXDTUROSFEDID) {
    if (slot < 7)
      ddu = 773;
    else
      ddu = 774;
  }

  if (ros > 6 && tenDDU && ddu < 775)
    ddu += 5;

  return ddu;
}

int DTDataIntegrityTask::theROS(int slot, int link) {

  if (slot%6 == 5) return link+1;

  int ros = (link/24) + 3*(slot%6) - 2;
  return ros;
}

  

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
