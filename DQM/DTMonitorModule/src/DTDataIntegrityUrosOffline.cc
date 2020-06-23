/*
 * \file DTDataIntegrityUrosOffline.cc
 *
 * \author Javier Fernandez (Uni. Oviedo) 
 *
 */

#include "DQM/DTMonitorModule/interface/DTDataIntegrityUrosOffline.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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

DTDataIntegrityUrosOffline::DTDataIntegrityUrosOffline(const edm::ParameterSet& ps) : nevents(0) {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
      << "[DTDataIntegrityUrosOffline]: Constructor" << endl;

  fedToken = consumes<DTuROSFEDDataCollection>(ps.getParameter<InputTag>("dtFEDlabel"));
  FEDIDmin = FEDNumbering::MINDTUROSFEDID;
  FEDIDmax = FEDNumbering::MAXDTUROSFEDID;

  neventsFED = 0;
  neventsuROS = 0;

  fedIntegrityFolder = ps.getUntrackedParameter<string>("fedIntegrityFolder", "DT/FEDIntegrity");
}

DTDataIntegrityUrosOffline::~DTDataIntegrityUrosOffline() {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
      << "[DTDataIntegrityUrosOffline]: Destructor. Analyzed " << neventsFED << " events" << endl;
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
      << "[DTDataIntegrityUrosOffline]: postEndJob called!" << endl;
}

/*
  Folder Structure uROS (starting 2018):
  - 3 uROS Summary plots: Wheel-1/-2 (FED1369), Wheel0 (FED1370), Wheel+1/+2 (FED1371)
  - One folder for each FED
  - Inside each FED folder the uROSStatus histos, FED histos
  - One folder for each wheel and the corresponding ROSn folders
  - Inside each ROS folder the TDC and ROS errors histos, 24 Links/plot
*/

void DTDataIntegrityUrosOffline::bookHistograms(DQMStore::IBooker& ibooker,
                                                edm::Run const& iRun,
                                                edm::EventSetup const& iSetup) {
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
      << "[DTDataIntegrityUrosOffline]: postBeginJob" << endl;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
      << "[DTDataIntegrityUrosOffline] Get DQMStore service" << endl;

  // Loop over the DT FEDs

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
      << " FEDS: " << FEDIDmin << " to " << FEDIDmax << " in the RO" << endl;

  // book FED integrity histos
  bookHistos(ibooker, FEDIDmin, FEDIDmax);

  // static booking of the histograms

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
}

void DTDataIntegrityUrosOffline::bookHistos(DQMStore::IBooker& ibooker, const int fedMin, const int fedMax) {
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

void DTDataIntegrityUrosOffline::bookHistos(DQMStore::IBooker& ibooker, string folder, const int fed) {
  string wheel = "ZERO";
  if (fed == FEDIDmin)
    wheel = "NEG";
  else if (fed == FEDIDmax)
    wheel = "POS";
  string fed_s = to_string(fed);
  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
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
  }

  // uROS Histograms
  if (folder == "FED") {  // The summary of the error of the ROS on the same FED
    ibooker.setCurrentFolder(topFolder(false));

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

void DTDataIntegrityUrosOffline::bookHistosROS(DQMStore::IBooker& ibooker, const int wheel, const int ros) {
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
  urosHistos[keyHisto] = ibooker.book2D(histoName, histoTitle, 5, 0, 5, 25, 0, 25);

  MonitorElement* histo = urosHistos[keyHisto];
  // uROS error bins
  // Placeholders for the moment
  histo->setBinLabel(1, "Error 1", 1);
  histo->setBinLabel(2, "Error 2", 1);
  histo->setBinLabel(3, "Error 3", 1);
  histo->setBinLabel(4, "Error 4", 1);
  histo->setBinLabel(5, "Not OKFlag", 1);
  for (int link = linkDown; link < (linkUp + 1); ++link) {
    string link_s = to_string(link);
    histo->setBinLabel(link + 1, "Link" + link_s, 2);
  }

}  //bookHistosROS

void DTDataIntegrityUrosOffline::bookHistosuROS(DQMStore::IBooker& ibooker, const int fed, const int uRos) {
  string fed_s = to_string(fed);
  string uRos_s = to_string(uRos);
  ibooker.setCurrentFolder(topFolder(false) + "FED" + fed_s + "/uROS" + uRos_s);
}

void DTDataIntegrityUrosOffline::processuROS(DTuROSROSData& data, int fed, int uRos) {
  neventsuROS++;

  LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
      << "[DTDataIntegrityUrosOffline]: " << neventsuROS << " events analyzed by processuROS" << endl;

  MonitorElement* uROSSummary = nullptr;
  uROSSummary = summaryHistos["uROSSummary"][fed];

  MonitorElement* uROSStatus = nullptr;
  uROSStatus = fedHistos["uROSStatus"][fed];

  if (!uROSSummary) {
    LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
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

  if (uRos > 2) {  //sectors 1-12

    uROSError0 = urosHistos[(uROSError)*1000 + (wheel + 2) * 100 + (ros - 1)];  //links 0-23
    uROSError1 = urosHistos[(uROSError)*1000 + (wheel + 2) * 100 + (ros)];      //links 24-47
    uROSError2 = urosHistos[(uROSError)*1000 + (wheel + 2) * 100 + (ros + 1)];  //links 48-71

    if ((!uROSError2) || (!uROSError1) || (!uROSError0)) {
      LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
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
              LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
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
    int error = data.geterror(icounter);
    int tdcError_ROSSummary = 0;
    int tdcError_ROSError = 0;

    if (error & 0x4000) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
          << " ROS " << uRos << " ROB " << link << " Internal fatal Error 4000 in TDC " << error << endl;

      tdcError_ROSSummary = 5;
      tdcError_ROSError = 5;

    } else if (error & 0x0249) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
          << " ROS " << uRos << " ROB " << link << " TDC FIFO overflow in TDC " << error << endl;

      tdcError_ROSSummary = 6;
      tdcError_ROSError = 6;

    } else if (error & 0x0492) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
          << " ROS " << uRos << " ROB " << link << " TDC L1 buffer overflow in TDC " << error << endl;

      tdcError_ROSSummary = 7;
      tdcError_ROSError = 7;

    } else if (error & 0x2000) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
          << " ROS " << uRos << " ROB " << link << " TDC L1A FIFO overflow in TDC " << error << endl;

      tdcError_ROSSummary = 8;
      tdcError_ROSError = 8;

    } else if (error & 0x0924) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
          << " uROS " << uRos << " ROB " << link << " TDC hit error in TDC " << error << endl;

      tdcError_ROSSummary = 9;
      tdcError_ROSError = 9;

    } else if (error & 0x1000) {
      LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
          << " uROS " << uRos << " ROB " << link << " TDC hit rejected in TDC " << error << endl;

      tdcError_ROSSummary = 10;
      tdcError_ROSError = 10;

    } else {
      LogWarning("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
          << " TDC error code not known " << error << endl;
    }

    if (uRos < 3) {
      ROSSummary->Fill(tdcError_ROSSummary, link + 1);  //link 0 = ROS 1
      int sc = 24;
      urosHistos[(uROSError)*1000 + (wheel + 2) * 100 + (link)]->Fill(tdcError_ROSError, sc);
    }       //uRos<3
    else {  //uRos>2
      if (link < 24) {
        ROSSummary->Fill(tdcError_ROSSummary, ros);
        uROSError0->Fill(tdcError_ROSError, link);
      } else if (link < 48) {
        ROSSummary->Fill(tdcError_ROSSummary, ros + 1);
        uROSError1->Fill(tdcError_ROSError, link - 23);
      } else if (link < 72) {
        ROSSummary->Fill(tdcError_ROSSummary, ros + 2);
        uROSError2->Fill(tdcError_ROSError, link - 47);
      }
    }  //uROS>2
  }    //loop on errors
}

void DTDataIntegrityUrosOffline::processFED(DTuROSFEDData& data, int fed) {
  neventsFED++;
  if (neventsFED % 1000 == 0)
    LogTrace("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
        << "[DTDataIntegrityUrosOffline]: " << neventsFED << " events analyzed by processFED" << endl;

  if (fed < FEDIDmin || fed > FEDIDmax)
    return;

  hFEDEntry->Fill(fed);

  //1D HISTOS: EVENT LENGHT from trailer
  int fedEvtLength = data.getevtlgth() * 8;  //1 word = 8 bytes
  //   if(fedEvtLength > 16000) fedEvtLength = 16000; // overflow bin
  fedHistos["EventLength"][fed]->Fill(fedEvtLength);
}

std::string DTDataIntegrityUrosOffline::topFolder(bool isFEDIntegrity) const {
  string folder = "DT/00-DataIntegrity/";

  return folder;
}

void DTDataIntegrityUrosOffline::analyze(const edm::Event& e, const edm::EventSetup& c) {
  nevents++;
  nEventMonitor->Fill(nevents);

  LogTrace("DTRawToDigi|TDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
      << "[DTDataIntegrityUrosOffline]: preProcessEvent" << endl;

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
        LogError("DTRawToDigi|DTDQM|DTMonitorModule|DTDataIntegrityUrosOffline")
            << "[DTDataIntegrityUrosOffline]: analyze, FED ID " << fed << " not expected." << endl;
        continue;
      }
      processFED(fedData, fed);

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
int DTDataIntegrityUrosOffline::theDDU(int crate, int slot, int link, bool tenDDU) {
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

int DTDataIntegrityUrosOffline::theROS(int slot, int link) {
  if (slot % 6 == 5)
    return link + 1;

  int ros = (link / 24) + 3 * (slot % 6) - 2;
  return ros;
}

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
