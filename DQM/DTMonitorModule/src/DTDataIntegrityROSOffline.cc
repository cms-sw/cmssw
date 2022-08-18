/*
 * \file DTDataIntegrityROSOffline.cc
 *
 * \author M. Zanetti (INFN Padova), S. Bolognesi (INFN Torino), G. Cerminara (INFN Torino)
 *
 */

#include "DQM/DTMonitorModule/interface/DTDataIntegrityROSOffline.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DTDigi/interface/DTControlData.h"
#include "DataFormats/DTDigi/interface/DTDDUWords.h"
#include "DQMServices/Core/interface/DQMStore.h"
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

DTDataIntegrityROSOffline::DTDataIntegrityROSOffline(const edm::ParameterSet& ps) : nevents(0) {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
      << "[DTDataIntegrityROSOffline]: Constructor" << endl;

  dduToken = consumes<DTDDUCollection>(ps.getUntrackedParameter<InputTag>("dtDDULabel"));
  ros25Token = consumes<DTROS25Collection>(ps.getUntrackedParameter<InputTag>("dtROS25Label"));
  FEDIDmin = FEDNumbering::MINDTFEDID;
  FEDIDmax = FEDNumbering::MAXDTFEDID;

  neventsFED = 0;

  //   Plot quantities about SC
  getSCInfo = ps.getUntrackedParameter<bool>("getSCInfo", false);

  fedIntegrityFolder = ps.getUntrackedParameter<string>("fedIntegrityFolder", "DT/FEDIntegrity");
}

DTDataIntegrityROSOffline::~DTDataIntegrityROSOffline() {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
      << "[DTDataIntegrityROSOffline]: Destructor. Analyzed " << neventsFED << " events" << endl;
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
      << "[DTDataIntegrityROSOffline]: postEndJob called!" << endl;
}

/*
  Folder Structure (ROS Legacy):
  - One folder for each DDU, named FEDn
  - Inside each DDU folder the DDU histos and the ROSn folder
  - Inside each ROS folder the ROS histos and the ROBn folder
  - Inside each ROB folder one occupancy plot and the TimeBoxes
  with the chosen granularity (simply change the histo name)
*/

void DTDataIntegrityROSOffline::bookHistograms(DQMStore::IBooker& ibooker,
                                               edm::Run const& iRun,
                                               edm::EventSetup const& iSetup) {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
      << "[DTDataIntegrityROSOffline]: postBeginJob" << endl;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
      << "[DTDataIntegrityROSOffline] Get DQMStore service" << endl;

  // Loop over the DT FEDs

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
      << " FEDS: " << FEDIDmin << " to " << FEDIDmax << " in the RO" << endl;

  // book FED integrity histos
  bookHistos(ibooker, FEDIDmin, FEDIDmax);

  //Legacy ROS

  // static booking of the histograms

  for (int fed = FEDIDmin; fed <= FEDIDmax; ++fed) {  // loop over the FEDs in the readout
    DTROChainCoding code;
    code.setDDU(fed);
    bookHistos(ibooker, string("ROS_S"), code);

    bookHistos(ibooker, string("DDU"), code);

    for (int ros = 1; ros <= nROS; ++ros) {  // loop over all ROS
      code.setROS(ros);
      bookHistosROS25(ibooker, code);
    }
  }
}

void DTDataIntegrityROSOffline::bookHistos(DQMStore::IBooker& ibooker, const int fedMin, const int fedMax) {
  ibooker.setCurrentFolder("DT/EventInfo/Counters");
  nEventMonitor = ibooker.bookFloat("nProcessedEventsDataIntegrity");

  // Standard FED integrity histos
  ibooker.setCurrentFolder(topFolder(true));

  int nFED = (fedMax - fedMin) + 1;

  hFEDEntry = ibooker.book1D("FEDEntries", "# entries per DT FED", nFED, fedMin, fedMax + 1);

  hFEDFatal = ibooker.book1D("FEDFatal", "# fatal errors DT FED", nFED, fedMin, fedMax + 1);
  hFEDNonFatal = ibooker.book1D("FEDNonFatal", "# NON fatal errors DT FED", nFED, fedMin, fedMax + 1);

  ibooker.setCurrentFolder(topFolder(false));
  hTTSSummary = ibooker.book2D("TTSSummary", "Summary Status TTS", nFED, fedMin, fedMax + 1, 9, 1, 10);
  hTTSSummary->setAxisTitle("FED", 1);
  hTTSSummary->setBinLabel(1, "ROS PAF", 2);
  hTTSSummary->setBinLabel(2, "DDU PAF", 2);
  hTTSSummary->setBinLabel(3, "ROS PAF", 2);
  hTTSSummary->setBinLabel(4, "DDU PAF", 2);
  hTTSSummary->setBinLabel(5, "DDU Full", 2);
  hTTSSummary->setBinLabel(6, "L1A Mism.", 2);
  hTTSSummary->setBinLabel(7, "ROS Error", 2);
  hTTSSummary->setBinLabel(8, "BX Mism.", 2);
  hTTSSummary->setBinLabel(9, "DDU Logic Err.", 2);

  // bookkeeping of the histos
  hCorruptionSummary =
      ibooker.book2D("DataCorruptionSummary", "Data Corruption Sources", nFED, fedMin, fedMax + 1, 8, 1, 9);
  hCorruptionSummary->setAxisTitle("FED", 1);
  hCorruptionSummary->setBinLabel(1, "Miss Ch.", 2);
  hCorruptionSummary->setBinLabel(2, "ROS BX mism", 2);
  hCorruptionSummary->setBinLabel(3, "DDU BX mism", 2);
  hCorruptionSummary->setBinLabel(4, "ROS L1A mism", 2);
  hCorruptionSummary->setBinLabel(5, "Miss Payload", 2);
  hCorruptionSummary->setBinLabel(6, "FCRC bit", 2);
  hCorruptionSummary->setBinLabel(7, "Header check", 2);
  hCorruptionSummary->setBinLabel(8, "Trailer Check", 2);
}

void DTDataIntegrityROSOffline::bookHistos(DQMStore::IBooker& ibooker, string folder, DTROChainCoding code) {
  string dduID_s = to_string(code.getDDU());
  string rosID_s = to_string(code.getROS());
  string robID_s = to_string(code.getROB());
  int wheel = (code.getDDUID() - 770) % 5 - 2;
  string wheel_s = to_string(wheel);

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
      << " Booking histos for FED: " << code.getDDU() << " ROS: " << code.getROS() << " ROB: " << code.getROB()
      << " folder: " << folder << endl;

  string histoType;
  string histoName;
  string histoTitle;
  MonitorElement* histo = nullptr;

  // DDU Histograms
  if (folder == "DDU") {
    ibooker.setCurrentFolder(topFolder(false) + "FED" + dduID_s);

    histoType = "EventLength";
    histoName = "FED" + dduID_s + "_" + histoType;
    histoTitle = "Event Length (Bytes) FED " + dduID_s;
    (fedHistos[histoType])[code.getDDUID()] = ibooker.book1D(histoName, histoTitle, 501, 0, 16032);

    histoType = "ROSStatus";
    histoName = "FED" + dduID_s + "_" + histoType;
    (fedHistos[histoType])[code.getDDUID()] = ibooker.book2D(histoName, histoName, 12, 0, 12, 12, 0, 12);
    histo = (fedHistos[histoType])[code.getDDUID()];
    histo->setBinLabel(1, "ch.enabled", 1);
    histo->setBinLabel(2, "timeout", 1);
    histo->setBinLabel(3, "ev.trailer lost", 1);
    histo->setBinLabel(4, "opt.fiber lost", 1);
    histo->setBinLabel(5, "tlk.prop.error", 1);
    histo->setBinLabel(6, "tlk.pattern error", 1);
    histo->setBinLabel(7, "tlk.sign.lost", 1);
    histo->setBinLabel(8, "error from ROS", 1);
    histo->setBinLabel(9, "if ROS in events", 1);
    histo->setBinLabel(10, "Miss. Evt.", 1);
    histo->setBinLabel(11, "Evt. ID Mismatch", 1);
    histo->setBinLabel(12, "BX Mismatch", 1);

    histo->setBinLabel(1, "ROS 1", 2);
    histo->setBinLabel(2, "ROS 2", 2);
    histo->setBinLabel(3, "ROS 3", 2);
    histo->setBinLabel(4, "ROS 4", 2);
    histo->setBinLabel(5, "ROS 5", 2);
    histo->setBinLabel(6, "ROS 6", 2);
    histo->setBinLabel(7, "ROS 7", 2);
    histo->setBinLabel(8, "ROS 8", 2);
    histo->setBinLabel(9, "ROS 9", 2);
    histo->setBinLabel(10, "ROS 10", 2);
    histo->setBinLabel(11, "ROS 11", 2);
    histo->setBinLabel(12, "ROS 12", 2);
  }

  // ROS Histograms
  if (folder == "ROS_S") {  // The summary of the error of the ROS on the same FED
    ibooker.setCurrentFolder(topFolder(false));

    histoType = "ROSSummary";
    histoName = "FED" + dduID_s + "_ROSSummary";
    string histoTitle = "Summary Wheel" + wheel_s + " (FED " + dduID_s + ")";

    ((summaryHistos[histoType])[code.getDDUID()]) = ibooker.book2D(histoName, histoTitle, 20, 0, 20, 12, 1, 13);
    MonitorElement* histo = ((summaryHistos[histoType])[code.getDDUID()]);
    // ROS error bins
    histo->setBinLabel(1, "Link TimeOut", 1);
    histo->setBinLabel(2, "Ev.Id.Mis.", 1);
    histo->setBinLabel(3, "FIFO almost full", 1);
    histo->setBinLabel(4, "FIFO full", 1);
    histo->setBinLabel(5, "CEROS timeout", 1);
    histo->setBinLabel(6, "Max. wds", 1);
    histo->setBinLabel(7, "WO L1A FIFO", 1);
    histo->setBinLabel(8, "TDC parity err.", 1);
    histo->setBinLabel(9, "BX ID Mis.", 1);
    histo->setBinLabel(10, "TXP", 1);
    histo->setBinLabel(11, "L1A almost full", 1);
    histo->setBinLabel(12, "Ch. blocked", 1);
    histo->setBinLabel(13, "Ev. Id. Mis.", 1);
    histo->setBinLabel(14, "CEROS blocked", 1);
    // TDC error bins
    histo->setBinLabel(15, "TDC Fatal", 1);
    histo->setBinLabel(16, "TDC RO FIFO ov.", 1);
    histo->setBinLabel(17, "TDC L1 buf. ov.", 1);
    histo->setBinLabel(18, "TDC L1A FIFO ov.", 1);
    histo->setBinLabel(19, "TDC hit err.", 1);
    histo->setBinLabel(20, "TDC hit rej.", 1);

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

  if (folder == "ROS") {
    ibooker.setCurrentFolder(topFolder(false) + "FED" + dduID_s + "/" + folder + rosID_s);

    histoType = "ROSError";
    histoName = "FED" + dduID_s + "_" + folder + rosID_s + "_ROSError";
    histoTitle = histoName + " (ROBID error summary)";
    (rosHistos[histoType])[code.getROSID()] = ibooker.book2D(histoName, histoTitle, 11, 0, 11, 26, 0, 26);

    MonitorElement* histo = (rosHistos[histoType])[code.getROSID()];
    // ROS error bins
    histo->setBinLabel(1, "Link TimeOut", 1);
    histo->setBinLabel(2, "Ev.Id.Mis.", 1);
    histo->setBinLabel(3, "FIFO almost full", 1);
    histo->setBinLabel(4, "FIFO full", 1);
    histo->setBinLabel(5, "CEROS timeout", 1);
    histo->setBinLabel(6, "Max. wds", 1);
    histo->setBinLabel(7, "TDC parity err.", 1);
    histo->setBinLabel(8, "BX ID Mis.", 1);
    histo->setBinLabel(9, "Ch. blocked", 1);
    histo->setBinLabel(10, "Ev. Id. Mis.", 1);
    histo->setBinLabel(11, "CEROS blocked", 1);

    histo->setBinLabel(1, "ROB0", 2);
    histo->setBinLabel(2, "ROB1", 2);
    histo->setBinLabel(3, "ROB2", 2);
    histo->setBinLabel(4, "ROB3", 2);
    histo->setBinLabel(5, "ROB4", 2);
    histo->setBinLabel(6, "ROB5", 2);
    histo->setBinLabel(7, "ROB6", 2);
    histo->setBinLabel(8, "ROB7", 2);
    histo->setBinLabel(9, "ROB8", 2);
    histo->setBinLabel(10, "ROB9", 2);
    histo->setBinLabel(11, "ROB10", 2);
    histo->setBinLabel(12, "ROB11", 2);
    histo->setBinLabel(13, "ROB12", 2);
    histo->setBinLabel(14, "ROB13", 2);
    histo->setBinLabel(15, "ROB14", 2);
    histo->setBinLabel(16, "ROB15", 2);
    histo->setBinLabel(17, "ROB16", 2);
    histo->setBinLabel(18, "ROB17", 2);
    histo->setBinLabel(19, "ROB18", 2);
    histo->setBinLabel(20, "ROB19", 2);
    histo->setBinLabel(21, "ROB20", 2);
    histo->setBinLabel(22, "ROB21", 2);
    histo->setBinLabel(23, "ROB22", 2);
    histo->setBinLabel(24, "ROB23", 2);
    histo->setBinLabel(25, "ROB24", 2);
    histo->setBinLabel(26, "SC", 2);
  }

  // SC Histograms
  if (folder == "SC") {
    // The plots are per wheel
    ibooker.setCurrentFolder(topFolder(false) + "FED" + dduID_s);

    // SC data Size
    histoType = "SCSizeVsROSSize";
    histoName = "FED" + dduID_s + "_SCSizeVsROSSize";
    histoTitle = "SC size vs SC (FED " + dduID_s + ")";
    rosHistos[histoType][code.getSCID()] = ibooker.book2D(histoName, histoTitle, 12, 1, 13, 51, -1, 50);
    rosHistos[histoType][code.getSCID()]->setAxisTitle("SC", 1);
  }
}

void DTDataIntegrityROSOffline::bookHistosROS25(DQMStore::IBooker& ibooker, DTROChainCoding code) {
  bookHistos(ibooker, string("ROS"), code);
}

void DTDataIntegrityROSOffline::processROS25(DTROS25Data& data, int ddu, int ros) {
  neventsROS++;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
      << "[DTDataIntegrityROSOffline]: " << neventsROS << " events analyzed by processROS25" << endl;

  // The ID of the RO board (used to map the histos)
  DTROChainCoding code;
  code.setDDU(ddu);
  code.setROS(ros);

  MonitorElement* ROSSummary = summaryHistos["ROSSummary"][code.getDDUID()];

  // Summary of all ROB errors
  MonitorElement* ROSError = nullptr;
  ROSError = rosHistos["ROSError"][code.getROSID()];

  if ((!ROSError)) {
    LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
        << "Trying to access non existing ME at ROSID " << code.getROSID() << std::endl;
    return;
  }

  // L1A ids to be checked against FED one
  rosL1AIdsPerFED[ddu].insert(data.getROSHeader().TTCEventCounter());

  // ROS errors

  // check for TPX errors
  if (data.getROSTrailer().TPX() != 0) {
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
        << " TXP error en ROS " << code.getROS() << endl;
    ROSSummary->Fill(9, code.getROS());
  }

  // L1 Buffer almost full (non-critical error!)
  if (data.getROSTrailer().l1AFifoOccupancy() > 31) {
    ROSSummary->Fill(10, code.getROS());
  }

  for (vector<DTROSErrorWord>::const_iterator error_it = data.getROSErrors().begin();
       error_it != data.getROSErrors().end();
       error_it++) {  // Loop over ROS error words

    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
        << " Error in ROS " << code.getROS() << " ROB Id " << (*error_it).robID() << " Error type "
        << (*error_it).errorType() << endl;

    // Fill the ROSSummary (1 per FED) histo
    ROSSummary->Fill((*error_it).errorType(), code.getROS());
    if ((*error_it).errorType() <= 11) {  // set error flag
      eventErrorFlag = true;
    }

    // Fill the ROB Summary (1 per ROS) histo
    if ((*error_it).robID() != 31) {
      ROSError->Fill((*error_it).errorType(), (*error_it).robID());
    } else if ((*error_it).errorType() == 4) {
      vector<int> channelBins;
      channelsInROS((*error_it).cerosID(), channelBins);
      vector<int>::const_iterator channelIt = channelBins.begin();
      vector<int>::const_iterator channelEnd = channelBins.end();
      for (; channelIt != channelEnd; ++channelIt) {
        ROSError->Fill(4, (*channelIt));
      }
    }
  }

  int ROSDebug_BunchNumber = -1;

  for (vector<DTROSDebugWord>::const_iterator debug_it = data.getROSDebugs().begin();
       debug_it != data.getROSDebugs().end();
       debug_it++) {  // Loop over ROS debug words

    int debugROSSummary = 0;
    int debugROSError = 0;
    vector<int> debugBins;
    bool hasEvIdMis = false;
    vector<int> evIdMisBins;

    if ((*debug_it).debugType() == 0) {
      ROSDebug_BunchNumber = (*debug_it).debugMessage();
    } else if ((*debug_it).debugType() == 1) {
      // not used
      // ROSDebug_BcntResCntLow = (*debug_it).debugMessage();
    } else if ((*debug_it).debugType() == 2) {
      // not used
      // ROSDebug_BcntResCntHigh = (*debug_it).debugMessage();
    } else if ((*debug_it).debugType() == 3) {
      if ((*debug_it).dontRead()) {
        debugROSSummary = 11;
        debugROSError = 8;
        channelsInCEROS((*debug_it).cerosIdCerosStatus(), (*debug_it).dontRead(), debugBins);
      }
      if ((*debug_it).evIdMis()) {
        hasEvIdMis = true;
        channelsInCEROS((*debug_it).cerosIdCerosStatus(), (*debug_it).evIdMis(), evIdMisBins);
      }
    } else if ((*debug_it).debugType() == 4 && (*debug_it).cerosIdRosStatus()) {
      debugROSSummary = 13;
      debugROSError = 10;
      channelsInROS((*debug_it).cerosIdRosStatus(), debugBins);
    }

    if (debugROSSummary) {
      ROSSummary->Fill(debugROSSummary, code.getROS());
      vector<int>::const_iterator channelIt = debugBins.begin();
      vector<int>::const_iterator channelEnd = debugBins.end();
      for (; channelIt != channelEnd; ++channelIt) {
        ROSError->Fill(debugROSError, (*channelIt));
      }
    }

    if (hasEvIdMis) {
      ROSSummary->Fill(12, code.getROS());
      vector<int>::const_iterator channelIt = evIdMisBins.begin();
      vector<int>::const_iterator channelEnd = evIdMisBins.end();
      for (; channelIt != channelEnd; ++channelIt) {
        ROSError->Fill(9, (*channelIt));
      }
    }
  }

  // ROB Group Header
  // Check the BX of the ROB headers against the BX of the ROS
  for (vector<DTROBHeader>::const_iterator rob_it = data.getROBHeaders().begin(); rob_it != data.getROBHeaders().end();
       rob_it++) {  // loop over ROB headers

    code.setROB((*rob_it).first);
    DTROBHeaderWord robheader = (*rob_it).second;

    rosBxIdsPerFED[ddu].insert(ROSDebug_BunchNumber);

    if (robheader.bunchID() != ROSDebug_BunchNumber) {
      // fill ROS Summary plot
      ROSSummary->Fill(8, code.getROS());
      eventErrorFlag = true;

      // fill ROB Summary plot for that particular ROS
      ROSError->Fill(7, robheader.robID());
    }
  }

  // TDC Data
  for (vector<DTTDCData>::const_iterator tdc_it = data.getTDCData().begin(); tdc_it != data.getTDCData().end();
       tdc_it++) {  // loop over TDC data

    DTTDCMeasurementWord tdcDatum = (*tdc_it).second;

    if (tdcDatum.PC() != 0) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
          << " PC error in ROS " << code.getROS() << " TDC " << (*tdc_it).first << endl;
      //     fill ROS Summary plot
      ROSSummary->Fill(7, code.getROS());

      eventErrorFlag = true;

      // fill ROB Summary plot for that particular ROS
      ROSError->Fill(6, (*tdc_it).first);
    }
  }

  // TDC Error
  for (vector<DTTDCError>::const_iterator tdc_it = data.getTDCError().begin(); tdc_it != data.getTDCError().end();
       tdc_it++) {  // loop over TDC errors

    code.setROB((*tdc_it).first);

    int tdcError_ROSSummary = 0;
    int tdcError_ROSError = 0;

    if (((*tdc_it).second).tdcError() & 0x4000) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
          << " ROS " << code.getROS() << " ROB " << code.getROB() << " Internal fatal Error 4000 in TDC "
          << (*tdc_it).first << endl;

      tdcError_ROSSummary = 14;
      tdcError_ROSError = 11;

    } else if (((*tdc_it).second).tdcError() & 0x0249) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
          << " ROS " << code.getROS() << " ROB " << code.getROB() << " TDC FIFO overflow in TDC " << (*tdc_it).first
          << endl;

      tdcError_ROSSummary = 15;
      tdcError_ROSError = 12;

    } else if (((*tdc_it).second).tdcError() & 0x0492) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
          << " ROS " << code.getROS() << " ROB " << code.getROB() << " TDC L1 buffer overflow in TDC "
          << (*tdc_it).first << endl;

      tdcError_ROSSummary = 16;
      tdcError_ROSError = 13;

    } else if (((*tdc_it).second).tdcError() & 0x2000) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
          << " ROS " << code.getROS() << " ROB " << code.getROB() << " TDC L1A FIFO overflow in TDC " << (*tdc_it).first
          << endl;

      tdcError_ROSSummary = 17;
      tdcError_ROSError = 14;

    } else if (((*tdc_it).second).tdcError() & 0x0924) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
          << " ROS " << code.getROS() << " ROB " << code.getROB() << " TDC hit error in TDC " << (*tdc_it).first
          << endl;

      tdcError_ROSSummary = 18;
      tdcError_ROSError = 15;

    } else if (((*tdc_it).second).tdcError() & 0x1000) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
          << " ROS " << code.getROS() << " ROB " << code.getROB() << " TDC hit rejected in TDC " << (*tdc_it).first
          << endl;

      tdcError_ROSSummary = 19;
      tdcError_ROSError = 16;

    } else {
      LogWarning("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
          << " TDC error code not known " << ((*tdc_it).second).tdcError() << endl;
    }

    ROSSummary->Fill(tdcError_ROSSummary, code.getROS());

    if (tdcError_ROSSummary <= 15) {
      eventErrorFlag = true;
    }

    ROSError->Fill(tdcError_ROSError, (*tdc_it).first);
  }
}

void DTDataIntegrityROSOffline::processFED(DTDDUData& data, const std::vector<DTROS25Data>& rosData, int ddu) {
  neventsFED++;
  if (neventsFED % 1000 == 0)
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
        << "[DTDataIntegrityROSOffline]: " << neventsFED << " events analyzed by processFED" << endl;

  DTROChainCoding code;
  code.setDDU(ddu);
  if (code.getDDUID() < FEDIDmin || code.getDDUID() > FEDIDmax)
    return;

  hFEDEntry->Fill(code.getDDUID());

  const FEDTrailer& trailer = data.getDDUTrailer();
  const FEDHeader& header = data.getDDUHeader();

  // check consistency of header and trailer
  if (!header.check()) {
    // error code 7
    hFEDFatal->Fill(code.getDDUID());
    hCorruptionSummary->Fill(code.getDDUID(), 7);
  }

  if (!trailer.check()) {
    // error code 8
    hFEDFatal->Fill(code.getDDUID());
    hCorruptionSummary->Fill(code.getDDUID(), 8);
  }

  // check CRC error bit set by DAQ before sending data on SLink
  if (data.crcErrorBit()) {
    // error code 6
    hFEDFatal->Fill(code.getDDUID());
    hCorruptionSummary->Fill(code.getDDUID(), 6);
  }

  const DTDDUSecondStatusWord& secondWord = data.getSecondStatusWord();

  //2D HISTO: ROS VS STATUS (8 BIT = 8 BIN) from 1st-2nd status words (9th BIN FROM LIST OF ROS in 2nd status word)
  MonitorElement* hROSStatus = fedHistos["ROSStatus"][code.getDDUID()];
  //1D HISTO: NUMBER OF ROS IN THE EVENTS from 2nd status word

  int rosList = secondWord.rosList();
  set<int> rosPositions;
  for (int i = 0; i < 12; i++) {
    if (rosList & 0x1) {
      rosPositions.insert(i);
      //9th BIN FROM LIST OF ROS in 2nd status word
      hROSStatus->Fill(8, i, 1);
    }
    rosList >>= 1;
  }

  int channel = 0;
  for (vector<DTDDUFirstStatusWord>::const_iterator fsw_it = data.getFirstStatusWord().begin();
       fsw_it != data.getFirstStatusWord().end();
       fsw_it++) {
    // assuming association one-to-one between DDU channel and ROS
    hROSStatus->Fill(0, channel, (*fsw_it).channelEnabled());
    hROSStatus->Fill(1, channel, (*fsw_it).timeout());
    hROSStatus->Fill(2, channel, (*fsw_it).eventTrailerLost());
    hROSStatus->Fill(3, channel, (*fsw_it).opticalFiberSignalLost());
    hROSStatus->Fill(4, channel, (*fsw_it).tlkPropagationError());
    hROSStatus->Fill(5, channel, (*fsw_it).tlkPatternError());
    hROSStatus->Fill(6, channel, (*fsw_it).tlkSignalLost());
    hROSStatus->Fill(7, channel, (*fsw_it).errorFromROS());
    // check that the enabled channel was also in the read-out
    if ((*fsw_it).channelEnabled() == 1 && rosPositions.find(channel) == rosPositions.end()) {
      hROSStatus->Fill(9, channel, 1);
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
  if ((rosBXIds.size() > 1 || rosBXIds.find(header.bxID()) == rosBXIds.end()) &&
      !rosBXIds.empty()) {  // in this case look for faulty ROSs
    for (vector<DTROS25Data>::const_iterator rosControlData = rosData.begin(); rosControlData != rosData.end();
         ++rosControlData) {  // loop over the ROS data
      for (vector<DTROSDebugWord>::const_iterator debug_it = (*rosControlData).getROSDebugs().begin();
           debug_it != (*rosControlData).getROSDebugs().end();
           debug_it++) {                                                                    // Loop over ROS debug words
        if ((*debug_it).debugType() == 0 && (*debug_it).debugMessage() != header.bxID()) {  // check the BX
          int ros = (*rosControlData).getROSID();
          // fill the error bin
          hROSStatus->Fill(11, ros - 1);
          // error code 2
          hFEDFatal->Fill(code.getDDUID());
          hCorruptionSummary->Fill(code.getDDUID(), 2);
        }
      }
    }
  }

  // check the BX ID against other FEDs
  fedBXIds.insert(header.bxID());
  if (fedBXIds.size() != 1) {
    LogWarning("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityROSOffline")
        << "ERROR: FED " << ddu << " BX ID different from other feds: " << header.bxID() << endl;
    // error code 3
    hFEDFatal->Fill(code.getDDUID());
    hCorruptionSummary->Fill(code.getDDUID(), 3);
  }

  // check the L1A ID against the ROSs
  set<int> rosL1AIds = rosL1AIdsPerFED[ddu];
  if ((rosL1AIds.size() > 1 || rosL1AIds.find(header.lvl1ID() - 1) == rosL1AIds.end()) &&
      !rosL1AIds.empty()) {  // in this case look for faulty ROSs
    //If L1A_ID error identify which ROS has wrong L1A
    for (vector<DTROS25Data>::const_iterator rosControlData = rosData.begin(); rosControlData != rosData.end();
         rosControlData++) {  // loop over the ROS data
      unsigned int ROSHeader_TTCCount =
          ((*rosControlData).getROSHeader().TTCEventCounter() + 1) %
          0x1000000;  // fix comparison in case of last counting bin in ROS /first one in DDU
      if (ROSHeader_TTCCount != header.lvl1ID()) {
        int ros = (*rosControlData).getROSID();
        hROSStatus->Fill(10, ros - 1);
        // error code 4
        hFEDFatal->Fill(code.getDDUID());
        hCorruptionSummary->Fill(code.getDDUID(), 4);
      }
    }
  }

  //1D HISTOS: EVENT LENGHT from trailer
  int fedEvtLength = trailer.fragmentLength() * 8;
  //   if(fedEvtLength > 16000) fedEvtLength = 16000; // overflow bin
  fedHistos["EventLength"][code.getDDUID()]->Fill(fedEvtLength);
}

// log number of times the payload of each fed is unpacked
void DTDataIntegrityROSOffline::fedEntry(int dduID) { hFEDEntry->Fill(dduID); }

// log number of times the payload of each fed is skipped (no ROS inside)
void DTDataIntegrityROSOffline::fedFatal(int dduID) { hFEDFatal->Fill(dduID); }

// log number of times the payload of each fed is partially skipped (some ROS skipped)
void DTDataIntegrityROSOffline::fedNonFatal(int dduID) { hFEDNonFatal->Fill(dduID); }

std::string DTDataIntegrityROSOffline::topFolder(bool isFEDIntegrity) const {
  string folder = "DT/00-DataIntegrity/";

  return folder;
}

void DTDataIntegrityROSOffline::channelsInCEROS(int cerosId, int chMask, vector<int>& channels) {
  for (int iCh = 0; iCh < 6; ++iCh) {
    if ((chMask >> iCh) & 0x1) {
      channels.push_back(cerosId * 6 + iCh);
    }
  }
  return;
}

void DTDataIntegrityROSOffline::channelsInROS(int cerosMask, vector<int>& channels) {
  for (int iCeros = 0; iCeros < 5; ++iCeros) {
    if ((cerosMask >> iCeros) & 0x1) {
      for (int iCh = 0; iCh < 6; ++iCh) {
        channels.push_back(iCeros * 6 + iCh);
      }
    }
  }
  return;
}

void DTDataIntegrityROSOffline::analyze(const edm::Event& e, const edm::EventSetup& c) {
  nevents++;
  nEventMonitor->Fill(nevents);

  LogTrace("DTRawToDigi|TDQM|DTMonitorModule|DTDataIntegrityROSOffline")
      << "[DTDataIntegrityROSOffline]: preProcessEvent" << endl;

  //Legacy ROS
  // clear the set of BXids from the ROSs
  for (map<int, set<int> >::iterator rosBxIds = rosBxIdsPerFED.begin(); rosBxIds != rosBxIdsPerFED.end(); ++rosBxIds) {
    (*rosBxIds).second.clear();
  }

  fedBXIds.clear();

  for (map<int, set<int> >::iterator rosL1AIds = rosL1AIdsPerFED.begin(); rosL1AIds != rosL1AIdsPerFED.end();
       ++rosL1AIds) {
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
  if (dduProduct.isValid() && ros25Product.isValid()) {
    for (unsigned int i = 0; i < dduProduct->size(); ++i) {
      dduData = dduProduct->at(i);
      ros25Data = ros25Product->at(i);
      FEDHeader header = dduData.getDDUHeader();
      int id = header.sourceID();
      if (id > FEDIDmax || id < FEDIDmin)
        continue;  //SIM uses extra FEDs not monitored

      processFED(dduData, ros25Data, id);
      for (unsigned int j = 0; j < ros25Data.size(); ++j) {
        int rosid = j + 1;
        processROS25(ros25Data[j], id, rosid);
      }
    }
  }
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
