/*
 * \file DTDataIntegrityTask.cc
 *
 * Class for DT Data Integrity
 * at Online DQM (Single Thread)
 * expected to monitor uROS
 * Class with MEs vs Time/LS
 *
 * \author Javier Fernandez (Uni. Oviedo) 
 *
 */

#include "DQM/DTMonitorModule/interface/DTDataIntegrityTask.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
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
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: Constructor" << endl;

  fedToken = consumes<DTuROSFEDDataCollection>(ps.getParameter<InputTag>("dtFEDlabel"));
  FEDIDmin = FEDNumbering::MINDTUROSFEDID;
  FEDIDmax = FEDNumbering::MAXDTUROSFEDID;

  neventsFED = 0;
  neventsuROS = 0;

  fedIntegrityFolder = ps.getUntrackedParameter<string>("fedIntegrityFolder", "DT/FEDIntegrity");

  string processingMode = ps.getUntrackedParameter<string>("processingMode", "Online");

  // processing mode flag to select plots to be produced and basedirs CB vedi se farlo meglio...
  if (processingMode == "Online") {
    mode = 0;
  } else if (processingMode == "SM") {
    mode = 1;
  } else if (processingMode == "Offline") {
    throw cms::Exception("WrongParameter") << "[DTDataIntegrityTask]: processingMode :" << processingMode
                                           << " invalid! Must be Online, SM or HLT !" << endl
                                           << " Offline mode is covered on DTDataIntegrityuROSOffline" << endl;
  } else if (processingMode == "HLT") {
    mode = 3;
  } else {
    throw cms::Exception("MissingParameter") << "[DTDataIntegrityTask]: processingMode :" << processingMode
                                             << " invalid! Must be Online, SM, Offline or HLT !" << endl;
  }
}

DTDataIntegrityTask::~DTDataIntegrityTask() {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << "[DTDataIntegrityTask]: Destructor. Analyzed " << neventsFED << " events" << endl;
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << "[DTDataIntegrityTask]: postEndJob called!" << endl;
}

/*
  Folder Structure uROS:
  - 3 uROS Summary plots: Wheel-1/-2 (FED1369), Wheel0 (FED1370), Wheel+1/+2 (FED1371)
  - One folder for each FED
  - Inside each FED folder the uROSStatus histos, FED histos
  - One folder for each wheel and the corresponding ROSn folders
  - Inside each ROS folder the TDC and ROS errors histos, 24 Links/plot
*/

void DTDataIntegrityTask::bookHistograms(DQMStore::IBooker& ibooker,
                                         edm::Run const& iRun,
                                         edm::EventSetup const& iSetup) {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: postBeginJob" << endl;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << "[DTDataIntegrityTask] Get DQMStore service" << endl;

  // Loop over the DT FEDs

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << " FEDS: " << FEDIDmin << " to " << FEDIDmax << " in the RO" << endl;

  // book FED integrity histos
  bookHistos(ibooker, FEDIDmin, FEDIDmax);

  // static booking of the histograms

  if (mode == 0) {
    for (int fed = FEDIDmin; fed <= FEDIDmax; ++fed) {  // loop over the FEDs in the readout

      bookHistos(ibooker, string("FED"), fed);

      bookHistos(ibooker, string("CRATE"), fed);

      for (int uRos = 1; uRos <= NuROS; ++uRos) {  // loop over all ROS
        bookHistosuROS(ibooker, fed, uRos);
      }
    }

    for (int wheel = -2; wheel < 3; ++wheel) {
      for (int ros = 1; ros <= NuROS; ++ros) {  // loop over all ROS
        bookHistosROS(ibooker, wheel, ros);
      }
    }

  }  //Not in HLT or SM mode
}

void DTDataIntegrityTask::bookHistos(DQMStore::IBooker& ibooker, const int fedMin, const int fedMax) {
  ibooker.setCurrentFolder("DT/EventInfo/Counters");
  nEventMonitor = ibooker.bookFloat("nProcessedEventsDataIntegrity");

  // Standard FED integrity histos
  ibooker.setCurrentFolder(topFolder(true));

  int nFED = (fedMax - fedMin) + 1;

  hFEDEntry = ibooker.book1D("FEDEntries", "# entries per DT FED", nFED, fedMin, fedMax + 1);

  string histoType = "ROSSummary";
  for (int wheel = -2; wheel < 3; ++wheel) {
    string wheel_s = to_string(wheel);
    string histoName = "ROSSummary_W" + wheel_s;
    string fed_s = to_string(FEDIDmin + 1);  //3 FEDs from 2018 onwards
    if (wheel < 0)
      fed_s = to_string(FEDIDmin);
    else if (wheel > 0)
      fed_s = to_string(FEDIDmax);
    string histoTitle = "Summary Wheel" + wheel_s + " (FED " + fed_s + ")";

    ((summaryHistos[histoType])[wheel]) = ibooker.book2D(histoName, histoTitle, 11, 0, 11, 12, 1, 13);
    MonitorElement* histo = ((summaryHistos[histoType])[wheel]);
    histo->setBinLabel(1, "Error 1", 1);
    histo->setBinLabel(2, "Error 2", 1);
    histo->setBinLabel(3, "Error 3", 1);
    histo->setBinLabel(4, "Error 4", 1);
    histo->setBinLabel(5, "Not OKflag", 1);
    // TDC error bins
    histo->setBinLabel(6, "TDC Fatal", 1);
    histo->setBinLabel(7, "TDC RO FIFO ov.", 1);
    histo->setBinLabel(8, "TDC L1 buf. ov.", 1);
    histo->setBinLabel(9, "TDC L1A FIFO ov.", 1);
    histo->setBinLabel(10, "TDC hit err.", 1);
    histo->setBinLabel(11, "TDC hit rej.", 1);

    histo->setBinLabel(1, "ROS1", 2);
    histo->setBinLabel(2, "ROS2", 2);
    histo->setBinLabel(3, "ROS3", 2);
    histo->setBinLabel(4, "ROS4", 2);
    histo->setBinLabel(5, "ROS5", 2);
    histo->setBinLabel(6, "ROS6", 2);
    histo->setBinLabel(7, "ROS7", 2);
    histo->setBinLabel(8, "ROS8", 2);
    histo->setBinLabel(9, "ROS9", 2);
    histo->setBinLabel(10, "ROS10", 2);
    histo->setBinLabel(11, "ROS11", 2);
    histo->setBinLabel(12, "ROS12", 2);
  }
}

void DTDataIntegrityTask::bookHistos(DQMStore::IBooker& ibooker, string folder, const int fed) {
  string wheel = "ZERO";
  if (fed == FEDIDmin)
    wheel = "NEG";
  else if (fed == FEDIDmax)
    wheel = "POS";
  string fed_s = to_string(fed);
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << " Booking histos for FED: " << fed_s << " folder: " << folder << endl;

  string histoType;
  string histoName;
  string histoTitle;
  MonitorElement* histo = nullptr;

  // Crate (old DDU) Histograms
  if (folder == "CRATE") {
    ibooker.setCurrentFolder(topFolder(false) + "FED" + fed_s);

    histoType = "EventLength";
    histoName = "FED" + fed_s + "_" + histoType;
    histoTitle = "Event Length (Bytes) FED " + fed_s;
    (fedHistos[histoType])[fed] = ibooker.book1D(histoName, histoTitle, 501, 0, 30000);

    if (mode == 3 || mode == 1)
      return;  //Avoid duplication of Info in FEDIntegrity_EvF

    histoType = "uROSStatus";
    histoName = "FED" + fed_s + "_" + histoType;
    (fedHistos[histoType])[fed] = ibooker.book2D(histoName, histoName, 12, 0, 12, 12, 1, 13);
    histo = (fedHistos[histoType])[fed];
    // only placeholders for the moment
    histo->setBinLabel(1, "Error G 1", 1);
    histo->setBinLabel(2, "Error G 2", 1);
    histo->setBinLabel(3, "Error G 3", 1);
    histo->setBinLabel(4, "Error G 4", 1);
    histo->setBinLabel(5, "Error G 5", 1);
    histo->setBinLabel(6, "Error G 6", 1);
    histo->setBinLabel(7, "Error G 7", 1);
    histo->setBinLabel(8, "Error G 8", 1);
    histo->setBinLabel(9, "Error G 9", 1);
    histo->setBinLabel(10, "Error G 10", 1);
    histo->setBinLabel(11, "Error G 11", 1);
    histo->setBinLabel(12, "Error G 12", 1);

    histo->setBinLabel(1, "uROS 1", 2);
    histo->setBinLabel(2, "uROS 2", 2);
    histo->setBinLabel(3, "uROS 3", 2);
    histo->setBinLabel(4, "uROS 4", 2);
    histo->setBinLabel(5, "uROS 5", 2);
    histo->setBinLabel(6, "uROS 6", 2);
    histo->setBinLabel(7, "uROS 7", 2);
    histo->setBinLabel(8, "uROS 8", 2);
    histo->setBinLabel(9, "uROS 9", 2);
    histo->setBinLabel(10, "uROS 10", 2);
    histo->setBinLabel(11, "uROS 11", 2);
    histo->setBinLabel(12, "uROS 12", 2);

    if (mode > 0)
      return;  //Info for Online only

    histoType = "FEDAvgEvLengthvsLumi";
    histoName = "FED" + fed_s + "_" + histoType;
    histoTitle = "Avg Event Length (Bytes) vs LumiSec FED " + fed_s;
    (fedTimeHistos[histoType])[fed] = new DTTimeEvolutionHisto(ibooker, histoName, histoTitle, 200, 10, true, 0);

    histoType = "TTSValues";
    histoName = "FED" + fed_s + "_" + histoType;
    (fedHistos[histoType])[fed] = ibooker.book1D(histoName, histoName, 8, 0, 8);
    histo = (fedHistos[histoType])[fed];
    histo->setBinLabel(1, "Disconnected", 1);
    histo->setBinLabel(2, "Overflow Warning ", 1);
    histo->setBinLabel(3, "Out of synch", 1);
    histo->setBinLabel(4, "Busy", 1);
    histo->setBinLabel(5, "Ready", 1);
    histo->setBinLabel(6, "Error", 1);
    histo->setBinLabel(7, "Disconnected", 1);
    histo->setBinLabel(8, "Unknown", 1);

    histoType = "uROSList";
    histoName = "FED" + fed_s + "_" + histoType;
    histoTitle = "# of uROS in the FED payload (FED" + fed_s + ")";
    (fedHistos[histoType])[fed] = ibooker.book1D(histoName, histoTitle, 13, 0, 13);

    histoType = "BXID";
    histoName = "FED" + fed_s + "_BXID";
    histoTitle = "Distrib. BX ID (FED" + fed_s + ")";
    (fedHistos[histoType])[fed] = ibooker.book1D(histoName, histoTitle, 3600, 0, 3600);
  }

  // uROS Histograms
  if (folder == "FED") {  // The summary of the error of the ROS on the same FED
    ibooker.setCurrentFolder(topFolder(false));

    if (mode == 3 || mode == 1)
      return;  //Avoid duplication of Info in FEDIntegrity_EvF

    histoType = "uROSSummary";
    histoName = "FED" + fed_s + "_uROSSummary";
    string histoTitle = "Summary Wheel" + wheel + " (FED " + fed_s + ")";

    ((summaryHistos[histoType])[fed]) = ibooker.book2D(histoName, histoTitle, 12, 0, 12, 12, 1, 13);
    MonitorElement* histo = ((summaryHistos[histoType])[fed]);
    // ROS error bins
    // Placeholders for Global Errors for the moment
    histo->setBinLabel(1, "Error G 1", 1);
    histo->setBinLabel(2, "Error G 2", 1);
    histo->setBinLabel(3, "Error G 3", 1);
    histo->setBinLabel(4, "Error G 4", 1);
    histo->setBinLabel(5, "Error G 5", 1);
    histo->setBinLabel(6, "Error G 6", 1);
    histo->setBinLabel(7, "Error G 7", 1);
    histo->setBinLabel(8, "Error G 8", 1);
    histo->setBinLabel(9, "Error G 9", 1);
    histo->setBinLabel(10, "Error G 10", 1);
    histo->setBinLabel(11, "Error G 11", 1);
    histo->setBinLabel(12, "Error G 12", 1);

    histo->setBinLabel(1, "uROS1", 2);
    histo->setBinLabel(2, "uROS2", 2);
    histo->setBinLabel(3, "uROS3", 2);
    histo->setBinLabel(4, "uROS4", 2);
    histo->setBinLabel(5, "uROS5", 2);
    histo->setBinLabel(6, "uROS6", 2);
    histo->setBinLabel(7, "uROS7", 2);
    histo->setBinLabel(8, "uROS8", 2);
    histo->setBinLabel(9, "uROS9", 2);
    histo->setBinLabel(10, "uROS10", 2);
    histo->setBinLabel(11, "uROS11", 2);
    histo->setBinLabel(12, "uROS12", 2);
  }
}

void DTDataIntegrityTask::bookHistosROS(DQMStore::IBooker& ibooker, const int wheel, const int ros) {
  string wheel_s = to_string(wheel);
  string ros_s = to_string(ros);
  ibooker.setCurrentFolder(topFolder(false) + "Wheel" + wheel_s + "/ROS" + ros_s);

  string histoType = "ROSError";
  int linkDown = 0;
  string linkDown_s = to_string(linkDown);
  int linkUp = linkDown + 24;
  string linkUp_s = to_string(linkUp);
  string histoName = "W" + wheel_s + "_" + "ROS" + ros_s + "_" + histoType;
  string histoTitle = histoName + " (Link " + linkDown_s + "-" + linkUp_s + " error summary)";
  unsigned int keyHisto = (uROSError)*1000 + (wheel + 2) * 100 + (ros - 1);
  if (mode < 1)  // Online only
    urosHistos[keyHisto] = ibooker.book2D(histoName, histoTitle, 11, 0, 11, 25, 0, 25);
  else
    urosHistos[keyHisto] = ibooker.book2D(histoName, histoTitle, 5, 0, 5, 25, 0, 25);

  MonitorElement* histo = urosHistos[keyHisto];
  // uROS error bins
  // Placeholders for the moment
  histo->setBinLabel(1, "Error 1", 1);
  histo->setBinLabel(2, "Error 2", 1);
  histo->setBinLabel(3, "Error 3", 1);
  histo->setBinLabel(4, "Error 4", 1);
  histo->setBinLabel(5, "Not OKFlag", 1);
  if (mode < 1) {  //Online only
                   // TDC error bins
    histo->setBinLabel(6, "TDC Fatal", 1);
    histo->setBinLabel(7, "TDC RO FIFO ov.", 1);
    histo->setBinLabel(8, "TDC L1 buf. ov.", 1);
    histo->setBinLabel(9, "TDC L1A FIFO ov.", 1);
    histo->setBinLabel(10, "TDC hit err.", 1);
    histo->setBinLabel(11, "TDC hit rej.", 1);
  }
  for (int link = linkDown; link < (linkUp + 1); ++link) {
    string link_s = to_string(link);
    histo->setBinLabel(link + 1, "Link" + link_s, 2);
  }

  if (mode > 1)
    return;

  histoType = "TDCError";
  linkDown = 0;
  linkDown_s = to_string(linkDown);
  linkUp = linkDown + 24;
  linkUp_s = to_string(linkUp);
  histoName = "W" + wheel_s + "_" + "ROS" + ros_s + "_" + histoType;
  histoTitle = histoName + " (Link " + linkDown_s + "-" + linkUp_s + " error summary)";
  keyHisto = (TDCError)*1000 + (wheel + 2) * 100 + (ros - 1);
  urosHistos[keyHisto] = ibooker.book2D(histoName, histoTitle, 24, 0, 24, 25, 0, 25);
  histo = urosHistos[keyHisto];
  // TDC error bins
  histo->setBinLabel(1, "Fatal", 1);
  histo->setBinLabel(2, "RO FIFO ov.", 1);
  histo->setBinLabel(3, "L1 buf. ov.", 1);
  histo->setBinLabel(4, "L1A FIFO ov.", 1);
  histo->setBinLabel(5, "hit err.", 1);
  histo->setBinLabel(6, "hit rej.", 1);
  histo->setBinLabel(7, "Fatal", 1);
  histo->setBinLabel(8, "RO FIFO ov.", 1);
  histo->setBinLabel(9, "L1 buf. ov.", 1);
  histo->setBinLabel(10, "L1A FIFO ov.", 1);
  histo->setBinLabel(11, "hit err.", 1);
  histo->setBinLabel(12, "hit rej.", 1);
  histo->setBinLabel(13, "Fatal", 1);
  histo->setBinLabel(14, "RO FIFO ov.", 1);
  histo->setBinLabel(15, "L1 buf. ov.", 1);
  histo->setBinLabel(16, "L1A FIFO ov.", 1);
  histo->setBinLabel(17, "hit err.", 1);
  histo->setBinLabel(18, "hit rej.", 1);
  histo->setBinLabel(19, "Fatal", 1);
  histo->setBinLabel(20, "RO FIFO ov.", 1);
  histo->setBinLabel(21, "L1 buf. ov.", 1);
  histo->setBinLabel(22, "L1A FIFO ov.", 1);
  histo->setBinLabel(23, "hit err.", 1);
  histo->setBinLabel(24, "hit rej.", 1);

  for (int link = linkDown; link < (linkUp + 1); ++link) {
    string link_s = to_string(link);
    histo->setBinLabel(link + 1, "Link" + link_s, 2);
  }
}  //bookHistosROS

void DTDataIntegrityTask::bookHistosuROS(DQMStore::IBooker& ibooker, const int fed, const int uRos) {
  string fed_s = to_string(fed);
  string uRos_s = to_string(uRos);
  ibooker.setCurrentFolder(topFolder(false) + "FED" + fed_s + "/uROS" + uRos_s);

  if (mode > 1)
    return;

  string histoType = "uROSEventLength";
  string histoName = "FED" + fed_s + "_uROS" + uRos_s + "_" + "EventLength";
  string histoTitle = "Event Length (Bytes) FED " + fed_s + " uROS" + uRos_s;
  unsigned int keyHisto = (uROSEventLength)*1000 + (fed - FEDIDmin) * 100 + (uRos - 1);
  urosHistos[keyHisto] = ibooker.book1D(histoName, histoTitle, 101, 0, 5000);

  histoType = "uROSAvgEventLengthvsLumi";
  histoName = "FED" + fed_s + "_ROS" + uRos_s + "AvgEventLengthvsLumi";
  histoTitle = "Event Length (Bytes) FED " + fed_s + " ROS" + uRos_s;
  keyHisto = (fed - FEDIDmin) * 100 + (uRos - 1);
  urosTimeHistos[keyHisto] = new DTTimeEvolutionHisto(ibooker, histoName, histoTitle, 200, 10, true, 0);

  histoType = "TTSValues";
  histoName = "FED" + fed_s + "_" + "uROS" + uRos_s + "_" + histoType;
  keyHisto = TTSValues * 1000 + (fed - FEDIDmin) * 100 + (uRos - 1);
  urosHistos[keyHisto] = ibooker.book1D(histoName, histoName, 8, 0, 8);
  MonitorElement* histo = urosHistos[keyHisto];
  histo->setBinLabel(1, "Disconnected", 1);
  histo->setBinLabel(2, "Overflow Warning ", 1);
  histo->setBinLabel(3, "Out of synch", 1);
  histo->setBinLabel(4, "Busy", 1);
  histo->setBinLabel(5, "Ready", 1);
  histo->setBinLabel(6, "Error", 1);
  histo->setBinLabel(7, "Disconnected", 1);
  histo->setBinLabel(8, "Unknown", 1);
}

void DTDataIntegrityTask::processuROS(DTuROSROSData& data, int fed, int uRos) {
  neventsuROS++;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
      << "[DTDataIntegrityTask]: " << neventsuROS << " events analyzed by processuROS" << endl;

  if (mode == 3 || mode == 1)
    return;  //Avoid duplication of Info in FEDIntegrity_EvF

  MonitorElement* uROSSummary = nullptr;
  uROSSummary = summaryHistos["uROSSummary"][fed];

  MonitorElement* uROSStatus = nullptr;
  uROSStatus = fedHistos["uROSStatus"][fed];

  if (!uROSSummary) {
    LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << "Trying to access non existing ME at FED " << fed << std::endl;
    return;
  }

  unsigned int slotMap = (data.getboardId()) & 0xF;
  if (slotMap == 0)
    return;                      //prevention for Simulation empty uROS data
  int ros = theROS(slotMap, 0);  //first sector correspondign to link 0
  int ddu = theDDU(fed, slotMap, 0, false);
  int wheel = (ddu - 770) % 5 - 2;
  MonitorElement* ROSSummary = nullptr;
  ROSSummary = summaryHistos["ROSSummary"][wheel];

  // Summary of all Link errors
  MonitorElement* uROSError0 = nullptr;
  MonitorElement* uROSError1 = nullptr;
  MonitorElement* uROSError2 = nullptr;

  int errorX[5][12] = {{0}};  //5th is notOK flag

  if (mode <= 2) {
    if (uRos > 2) {  //sectors 1-12

      uROSError0 = urosHistos[(uROSError)*1000 + (wheel + 2) * 100 + (ros - 1)];  //links 0-23
      uROSError1 = urosHistos[(uROSError)*1000 + (wheel + 2) * 100 + (ros)];      //links 24-47
      uROSError2 = urosHistos[(uROSError)*1000 + (wheel + 2) * 100 + (ros + 1)];  //links 48-71

      if ((!uROSError2) || (!uROSError1) || (!uROSError0)) {
        LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
            << "Trying to access non existing ME at uROS " << uRos << std::endl;
        return;
      }

      // uROS errors
      for (unsigned int link = 0; link < 72; ++link) {
        for (unsigned int flag = 0; flag < 5; ++flag) {
          if ((data.getokxflag(link) >> flag) & 0x1) {  // Undefined Flag 1-4 64bits word for each MTP (12 channels)
            int value = flag;
            if (flag == 0)
              value = 5;  //move it to the 5th bin

            if (value > 0) {
              if (link < 24) {
                errorX[value - 1][ros - 1] += 1;
                uROSError0->Fill(value - 1, link);  //bins start at 0 despite labelin
              } else if (link < 48) {
                errorX[value - 1][ros] += 1;
                uROSError1->Fill(value - 1, link - 23);
              } else if (link < 72) {
                errorX[value - 1][ros + 1] += 1;
                uROSError2->Fill(value - 1, link - 47);
              }
            }  //value>0
          }    //flag value
        }      //loop on flags
      }        //loop on links
    }          //uROS>2

    else {  //uRos<3

      for (unsigned int link = 0; link < 12; ++link) {
        for (unsigned int flag = 0; flag < 5; ++flag) {
          if ((data.getokxflag(link) >> flag) & 0x1) {  // Undefined Flag 1-4 64bits word for each MTP (12 channels)
            int value = flag;
            int sc = 24;
            if (flag == 0)
              value = 5;  //move it to the 5th bin

            if (value > 0) {
              unsigned int keyHisto = (uROSError)*1000 + (wheel + 2) * 100 + link;  //ros -1 = link in this case
              uROSError0 = urosHistos[keyHisto];
              if (!uROSError0) {
                LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
                    << "Trying to access non existing ME at uROS " << uRos << std::endl;
                return;
              }
              errorX[value - 1][link] += 1;     // ros-1=link in this case
              uROSError0->Fill(value - 1, sc);  //bins start at 0 despite labeling, this is the old SC
            }
          }  //flag values
        }    //loop on flags
      }      //loop on links
    }        //else uRos<3

  }  //mode<=2

  // Fill the ROSSummary (1 per wheel) histo
  for (unsigned int iros = 0; iros < 12; ++iros) {
    for (unsigned int bin = 0; bin < 5; ++bin) {
      if (errorX[bin][iros] != 0)
        ROSSummary->Fill(bin, iros + 1);  //bins start at 1
    }
  }

  // Global Errors for uROS
  for (unsigned int flag = 4; flag < 16; ++flag) {
    if ((data.getuserWord() >> flag) & 0x1) {
      uROSSummary->Fill(flag - 4, uRos);
      uROSStatus->Fill(flag - 4, uRos);  //duplicated info?
    }
  }

  // ROS error
  for (unsigned int icounter = 0; icounter < data.geterrors().size(); ++icounter) {
    int link = data.geterrorROBID(icounter);
    int tdc = data.geterrorTDCID(icounter);
    int error = data.geterror(icounter);
    int tdcError_ROSSummary = 0;
    int tdcError_ROSError = 0;
    int tdcError_TDCHisto = 0;

    if (error & 0x4000) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
          << " ROS " << uRos << " ROB " << link << " Internal fatal Error 4000 in TDC " << error << endl;

      tdcError_ROSSummary = 5;
      tdcError_ROSError = 5;
      tdcError_TDCHisto = 0;

    } else if (error & 0x0249) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
          << " ROS " << uRos << " ROB " << link << " TDC FIFO overflow in TDC " << error << endl;

      tdcError_ROSSummary = 6;
      tdcError_ROSError = 6;
      tdcError_TDCHisto = 1;

    } else if (error & 0x0492) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
          << " ROS " << uRos << " ROB " << link << " TDC L1 buffer overflow in TDC " << error << endl;

      tdcError_ROSSummary = 7;
      tdcError_ROSError = 7;
      tdcError_TDCHisto = 2;

    } else if (error & 0x2000) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
          << " ROS " << uRos << " ROB " << link << " TDC L1A FIFO overflow in TDC " << error << endl;

      tdcError_ROSSummary = 8;
      tdcError_ROSError = 8;
      tdcError_TDCHisto = 3;

    } else if (error & 0x0924) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
          << " uROS " << uRos << " ROB " << link << " TDC hit error in TDC " << error << endl;

      tdcError_ROSSummary = 9;
      tdcError_ROSError = 9;
      tdcError_TDCHisto = 4;

    } else if (error & 0x1000) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
          << " uROS " << uRos << " ROB " << link << " TDC hit rejected in TDC " << error << endl;

      tdcError_ROSSummary = 10;
      tdcError_ROSError = 10;
      tdcError_TDCHisto = 5;

    } else {
      LogWarning("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
          << " TDC error code not known " << error << endl;
    }

    if (uRos < 3) {
      ROSSummary->Fill(tdcError_ROSSummary, link + 1);  //link 0 = ROS 1
      int sc = 24;
      if (mode <= 2) {
        urosHistos[(uROSError)*1000 + (wheel + 2) * 100 + (link)]->Fill(tdcError_ROSError, sc);
        if (mode <= 1)
          urosHistos[(TDCError)*1000 + (wheel + 2) * 100 + (link)]->Fill(tdcError_TDCHisto + 6 * (tdc - 1),
                                                                         sc);  // ros-1=link in this case
      }                                                                        //mode<=2
    }                                                                          //uRos<3
    else {                                                                     //uRos>2
      if (link < 24)
        ROSSummary->Fill(tdcError_ROSSummary, ros);
      else if (link < 48)
        ROSSummary->Fill(tdcError_ROSSummary, ros + 1);
      else if (link < 72)
        ROSSummary->Fill(tdcError_ROSSummary, ros + 2);

      if (mode <= 2) {
        if (link < 24)
          uROSError0->Fill(tdcError_ROSError, link);
        else if (link < 48)
          uROSError1->Fill(tdcError_ROSError, link - 23);
        else if (link < 72)
          uROSError2->Fill(tdcError_ROSError, link - 47);

        if (mode <= 1) {
          if (link < 24)
            urosHistos[(TDCError)*1000 + (wheel + 2) * 100 + (ros - 1)]->Fill(tdcError_TDCHisto + 6 * (tdc - 1), link);
          else if (link < 48)
            urosHistos[(TDCError)*1000 + (wheel + 2) * 100 + (ros)]->Fill(tdcError_TDCHisto + 6 * (tdc - 1), link - 23);
          else if (link < 72)
            urosHistos[(TDCError)*1000 + (wheel + 2) * 100 + (ros + 1)]->Fill(tdcError_TDCHisto + 6 * (tdc - 1),
                                                                              link - 47);

        }  //mode<=1
      }    //mode<=2
    }      //uROS>2
  }        //loop on errors

  // 1D histograms for TTS values per uROS
  int ttsCodeValue = -1;

  int value = (data.getuserWord() & 0xF);
  switch (value) {
    case 0: {  //disconnected
      ttsCodeValue = 0;
      break;
    }
    case 1: {  //warning overflow
      ttsCodeValue = 1;
      break;
    }
    case 2: {  //out of sinch
      ttsCodeValue = 2;
      break;
    }
    case 4: {  //busy
      ttsCodeValue = 3;
      break;
    }
    case 8: {  //ready
      ttsCodeValue = 4;
      break;
    }
    case 12: {  //error
      ttsCodeValue = 5;
      break;
    }
    case 15: {  //disconnected
      ttsCodeValue = 6;
      break;
    }
    default: {
      LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
          << "[DTDataIntegrityTask] FED User control: wrong TTS value " << value << " in FED " << fed << " uROS "
          << uRos << endl;
      ttsCodeValue = 7;
    }
  }
  if (mode < 1) {
    urosHistos[TTSValues * 1000 + (fed - FEDIDmin) * 100 + (uRos - 1)]->Fill(ttsCodeValue);

    // Plot the event length //NOHLT
    int uRosEventLength = (data.gettrailer() & 0xFFFFF) * 8;
    urosTimeHistos[(fed - FEDIDmin) * 100 + (uRos - 1)]->accumulateValueTimeSlot(uRosEventLength);

    if (uRosEventLength > 5000)
      uRosEventLength = 5000;
    urosHistos[uROSEventLength * 1000 + (fed - FEDIDmin) * 100 + (uRos - 1)]->Fill(uRosEventLength);
  }
}

void DTDataIntegrityTask::processFED(DTuROSFEDData& data, int fed) {
  neventsFED++;
  if (neventsFED % 1000 == 0)
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
        << "[DTDataIntegrityTask]: " << neventsFED << " events analyzed by processFED" << endl;

  if (fed < FEDIDmin || fed > FEDIDmax)
    return;

  hFEDEntry->Fill(fed);

  if (mode == 3 || mode == 1)
    return;  //Avoid duplication of Info in FEDIntegrity_EvF

  //1D HISTOS: EVENT LENGHT from trailer
  int fedEvtLength = data.getevtlgth() * 8;  //1 word = 8 bytes
  //   if(fedEvtLength > 16000) fedEvtLength = 16000; // overflow bin
  fedHistos["EventLength"][fed]->Fill(fedEvtLength);

  if (mode > 1)
    return;

  fedTimeHistos["FEDAvgEvLengthvsLumi"][fed]->accumulateValueTimeSlot(fedEvtLength);

  // fill the distribution of the BX ids
  fedHistos["BXID"][fed]->Fill(data.getBXId());

  // size of the list of ROS in the Read-Out
  fedHistos["uROSList"][fed]->Fill(data.getnslots());

  // Fill the status summary of the TTS

  //1D HISTO WITH TTS VALUES form trailer (7 bins = 7 values)

  int ttsCodeValue = -1;
  int value = data.getTTS();
  switch (value) {
    case 0: {  //disconnected
      ttsCodeValue = 0;
      break;
    }
    case 1: {  //warning overflow
      ttsCodeValue = 1;
      break;
    }
    case 2: {  //out of sinch
      ttsCodeValue = 2;
      break;
    }
    case 4: {  //busy
      ttsCodeValue = 3;
      break;
    }
    case 8: {  //ready
      ttsCodeValue = 4;
      break;
    }
    case 12: {  //error
      ttsCodeValue = 5;
      break;
    }
    case 15: {  //disconnected
      ttsCodeValue = 6;
      break;
    }
    default: {
      LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
          << "[DTDataIntegrityTask] FED TTS control: wrong TTS value " << value << " in FED " << fed << endl;
      ttsCodeValue = 7;
    }
  }
  if (mode < 1)
    fedHistos["TTSValues"][fed]->Fill(ttsCodeValue);
}

std::string DTDataIntegrityTask::topFolder(bool isFEDIntegrity) const {
  string folder = isFEDIntegrity ? fedIntegrityFolder : "DT/00-DataIntegrity/";

  if (mode == 0)
    folder = "DT/00-DataIntegrity/";  //Move everything from FEDIntegrity except for SM and HLT modes

  return folder;
}

void DTDataIntegrityTask::beginLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) {
  nEventsLS = 0;
}

void DTDataIntegrityTask::endLuminosityBlock(const edm::LuminosityBlock& ls, const edm::EventSetup& es) {
  int lumiBlock = ls.luminosityBlock();

  map<string, map<int, DTTimeEvolutionHisto*> >::iterator fedIt = fedTimeHistos.begin();
  map<string, map<int, DTTimeEvolutionHisto*> >::iterator fedEnd = fedTimeHistos.end();
  for (; fedIt != fedEnd; ++fedIt) {
    map<int, DTTimeEvolutionHisto*>::iterator histoIt = fedIt->second.begin();
    map<int, DTTimeEvolutionHisto*>::iterator histoEnd = fedIt->second.end();
    for (; histoIt != histoEnd; ++histoIt) {
      histoIt->second->updateTimeSlot(lumiBlock, nEventsLS);
    }
  }

  map<unsigned int, DTTimeEvolutionHisto*>::iterator urosIt = urosTimeHistos.begin();
  map<unsigned int, DTTimeEvolutionHisto*>::iterator urosEnd = urosTimeHistos.end();
  for (; urosIt != urosEnd; ++urosIt) {
    urosIt->second->updateTimeSlot(lumiBlock, nEventsLS);
  }
}

void DTDataIntegrityTask::analyze(const edm::Event& e, const edm::EventSetup& c) {
  nevents++;
  nEventMonitor->Fill(nevents);

  nEventsLS++;

  LogTrace("DTRawToDigi|TDQM|DTMonitorModule|DTDataIntegrityTask") << "[DTDataIntegrityTask]: preProcessEvent" << endl;

  // Digi collection
  edm::Handle<DTuROSFEDDataCollection> fedCol;
  e.getByToken(fedToken, fedCol);
  DTuROSFEDData fedData;
  DTuROSROSData urosData;

  if (fedCol.isValid()) {
    for (unsigned int j = 0; j < fedCol->size(); ++j) {
      fedData = fedCol->at(j);
      int fed = fedData.getfed();  //argument should be void
      if (fed > FEDIDmax || fed < FEDIDmin) {
        LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityTask")
            << "[DTDataIntegrityTask]: analyze, FED ID " << fed << " not expected." << endl;
        continue;
      }
      processFED(fedData, fed);

      if (mode == 3 || mode == 1)
        continue;  //Not needed for FEDIntegrity_EvF

      for (int slot = 1; slot <= DOCESLOTS; ++slot) {
        urosData = fedData.getuROS(slot);
        if (fedData.getslotsize(slot) == 0 || urosData.getslot() == -1)
          continue;
        processuROS(urosData, fed, slot);
      }
    }
  }
}

// Conversions
int DTDataIntegrityTask::theDDU(int crate, int slot, int link, bool tenDDU) {
  int ros = theROS(slot, link);

  int ddu = 772;
  //if (crate == 1368) { ddu = 775; }
  //Needed just in case this FED should be used due to fibers length

  if (crate == FEDNumbering::MINDTUROSFEDID) {
    if (slot < 7)
      ddu = 770;
    else
      ddu = 771;
  }

  if (crate == (FEDNumbering::MINDTUROSFEDID + 1)) {
    ddu = 772;
  }

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
  if (slot % 6 == 5)
    return link + 1;

  int ros = (link / 24) + 3 * (slot % 6) - 2;
  return ros;
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
