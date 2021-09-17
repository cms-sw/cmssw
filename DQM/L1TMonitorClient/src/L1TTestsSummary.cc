#include "DQM/L1TMonitorClient/interface/L1TTestsSummary.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <cstdio>
#include <sstream>
#include <cmath>
#include <vector>
#include <TMath.h>
#include <climits>
#include <TFile.h>
#include <TDirectory.h>
#include <TProfile.h>

using namespace std;
using namespace edm;

//____________________________________________________________________________
// Function: L1TTestsSummary
// Description: This is the constructor, basic variable initialization
// Inputs:
// * const edm::ParameterSet& ps = Parameter for this analyzer
//____________________________________________________________________________
L1TTestsSummary::L1TTestsSummary(const edm::ParameterSet &ps) {
  if (mVerbose) {
    cout << "[L1TTestsSummary:] Called constructor" << endl;
  }

  // Get parameters
  mParameters = ps;
  mVerbose = ps.getUntrackedParameter<bool>("verbose", true);
  mMonitorL1TRate = ps.getUntrackedParameter<bool>("MonitorL1TRate", true);
  mMonitorL1TSync = ps.getUntrackedParameter<bool>("MonitorL1TSync", true);
  mMonitorL1TOccupancy = ps.getUntrackedParameter<bool>("MonitorL1TOccupancy", true);

  mL1TRatePath = ps.getUntrackedParameter<string>("L1TRatePath", "L1T/L1TRate/Certification/");
  mL1TSyncPath = ps.getUntrackedParameter<string>("L1TSyncPath", "L1T/L1TSync/Certification/");
  mL1TOccupancyPath = ps.getUntrackedParameter<string>("L1TOccupancyPath", "L1T/L1TOccupancy/Certification/");
}

//____________________________________________________________________________
// Function: ~L1TTestsSummary
// Description: This is the destructor, basic variable deletion
//____________________________________________________________________________
L1TTestsSummary::~L1TTestsSummary() {
  if (mVerbose) {
    cout << "[L1TTestsSummary:] Called destructor" << endl;
  }
}

//____________________________________________________________________________
// Function: beginRun
// Description: This is will be run at the begining of each run
// Inputs:
// * const Run&        r       = Run information
// * const EventSetup& context = Event Setup information
//____________________________________________________________________________
void L1TTestsSummary::book(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  if (mVerbose) {
    cout << "[L1TTestsSummary:] Called beginRun" << endl;
  }

  int maxLS = 2500;

  if (mMonitorL1TRate) {
    if (mVerbose) {
      cout << "[L1TTestsSummary:] Initializing L1TRate Module Monitoring" << endl;
    }

    igetter.setCurrentFolder(mL1TRatePath);
    vector<string> histToMonitor = igetter.getMEs();
    int histLines = histToMonitor.size() + 1;

    ibooker.setCurrentFolder("L1T/L1TTestsSummary/");
    mL1TRateMonitor = ibooker.book2D(
        "RateQualitySummary", "L1T Rates Monitor Summary", maxLS, +0.5, double(maxLS) + 0.5, histLines, 0, histLines);
    mL1TRateMonitor->setAxisTitle("Lumi Section", 1);

    mL1TRateMonitor->setBinLabel(1, "Summary", 2);
    for (unsigned int i = 0; i < histToMonitor.size(); i++) {
      string name = igetter.get(mL1TRatePath + histToMonitor[i])->getTH1()->GetName();
      mL1TRateMonitor->setBinLabel(i + 2, name, 2);
    }
  }
  if (mMonitorL1TSync) {
    if (mVerbose) {
      cout << "[L1TTestsSummary:] Initializing L1TSync Module Monitoring" << endl;
    }

    igetter.setCurrentFolder(mL1TSyncPath);
    vector<string> histToMonitor = igetter.getMEs();
    int histLines = histToMonitor.size() + 1;

    ibooker.setCurrentFolder("L1T/L1TTestsSummary/");
    mL1TSyncMonitor = ibooker.book2D("SyncQualitySummary",
                                     "L1T Synchronization Monitor Summary",
                                     maxLS,
                                     0.5,
                                     double(maxLS) + 0.5,
                                     histLines,
                                     0,
                                     histLines);
    mL1TSyncMonitor->setAxisTitle("Lumi Section", 1);

    mL1TSyncMonitor->setBinLabel(1, "Summary", 2);
    for (unsigned int i = 0; i < histToMonitor.size(); i++) {
      string name = igetter.get(mL1TSyncPath + histToMonitor[i])->getTH1()->GetName();
      mL1TSyncMonitor->setBinLabel(i + 2, name, 2);
    }
  }
  if (mMonitorL1TOccupancy) {
    if (mVerbose) {
      cout << "[L1TTestsSummary:] Initializing L1TOccupancy Module Monitoring" << endl;
    }

    igetter.setCurrentFolder(mL1TOccupancyPath);
    vector<string> histToMonitor = igetter.getMEs();
    int histLines = histToMonitor.size() + 1;

    ibooker.setCurrentFolder("L1T/L1TTestsSummary/");
    mL1TOccupancyMonitor = ibooker.book2D(
        "OccupancySummary", "L1T Occupancy Monitor Summary", maxLS, +0.5, double(maxLS) + 0.5, histLines, 0, histLines);
    mL1TOccupancyMonitor->setAxisTitle("Lumi Section", 1);

    mL1TOccupancyMonitor->setBinLabel(1, "Summary", 2);
    for (unsigned int i = 0; i < histToMonitor.size(); i++) {
      string name = igetter.get(mL1TOccupancyPath + histToMonitor[i])->getTH1()->GetName();
      mL1TOccupancyMonitor->setBinLabel(i + 2, name, 2);
    }
  }

  //-> Making the summary of summaries
  int testsToMonitor = 1;
  if (mMonitorL1TRate) {
    testsToMonitor++;
  }
  if (mMonitorL1TSync) {
    testsToMonitor++;
  }
  if (mMonitorL1TOccupancy) {
    testsToMonitor++;
  }

  // Creating
  ibooker.setCurrentFolder("L1T/L1TTestsSummary/");
  mL1TSummary = ibooker.book2D(
      "L1TQualitySummary", "L1 Tests Summary", maxLS, +0.5, double(maxLS) + 0.5, testsToMonitor, 0, testsToMonitor);
  mL1TSummary->setAxisTitle("Lumi Section", 1);
  mL1TSummary->setBinLabel(1, "L1T Summary", 2);

  int it = 2;
  if (mMonitorL1TRate) {
    mL1TSummary->setBinLabel(it, "Rates", 2);
    binYRate = it;
    it++;
  }
  if (mMonitorL1TSync) {
    mL1TSummary->setBinLabel(it, "Synchronization", 2);
    binYSync = it;
    it++;
  }
  if (mMonitorL1TOccupancy) {
    mL1TSummary->setBinLabel(it, "Occupancy", 2);
    binYOccpancy = it;
  }
}

//____________________________________________________________________________
// Function: endRun
// Description: This is will be run at the end of each run
// Inputs:
// * const Run&        r       = Run information
// * const EventSetup& context = Event Setup information
//____________________________________________________________________________
void L1TTestsSummary::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  book(ibooker, igetter);

  if (mVerbose) {
    cout << "[L1TTestsSummary:] Called endRun()" << endl;
  }

  if (mMonitorL1TRate) {
    updateL1TRateMonitor(ibooker, igetter);
  }
  if (mMonitorL1TSync) {
    updateL1TSyncMonitor(ibooker, igetter);
  }
  if (mMonitorL1TOccupancy) {
    updateL1TOccupancyMonitor(ibooker, igetter);
  }
  updateL1TSummary(ibooker, igetter);
}

//____________________________________________________________________________
// Function: endLuminosityBlock
// Description: This is will be run at the end of each luminosity block
// Inputs:
// * const LuminosityBlock& lumiSeg = Luminosity Block information
// * const EventSetup&      context = Event Setup information
//____________________________________________________________________________
void L1TTestsSummary::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                                            DQMStore::IGetter &igetter,
                                            const edm::LuminosityBlock &lumiSeg,
                                            const edm::EventSetup &c) {
  int eventLS = lumiSeg.id().luminosityBlock();

  book(ibooker, igetter);

  mProcessedLS.push_back(eventLS);

  if (mVerbose) {
    cout << "[L1TTestsSummary:] Called endLuminosityBlock()" << endl;
    cout << "[L1TTestsSummary:] Lumisection: " << eventLS << endl;
  }

  if (mMonitorL1TRate) {
    updateL1TRateMonitor(ibooker, igetter);
  }
  if (mMonitorL1TSync) {
    updateL1TSyncMonitor(ibooker, igetter);
  }
  if (mMonitorL1TOccupancy) {
    updateL1TOccupancyMonitor(ibooker, igetter);
  }
  updateL1TSummary(ibooker, igetter);
}

//____________________________________________________________________________
// Function: updateL1TRateMonitor
// Description:
//____________________________________________________________________________
void L1TTestsSummary::updateL1TRateMonitor(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  igetter.setCurrentFolder(mL1TRatePath);
  vector<string> histToMonitor = igetter.getMEs();

  for (unsigned int i = 0; i < histToMonitor.size(); i++) {
    MonitorElement *me = igetter.get(mL1TRatePath + histToMonitor[i]);
    if (mVerbose) {
      cout << "[L1TTestsSummary:] Found ME: " << me->getTH1()->GetName() << endl;
    }

    const QReport *myQReport = me->getQReport("L1TRateTest");  //get QReport associated to your ME
    if (myQReport) {
      float qtresult = myQReport->getQTresult();          // get QT result value
      int qtstatus = myQReport->getStatus();              // get QT status value (see table below)
      const string &qtmessage = myQReport->getMessage();  // get the whole QT result message
      vector<DQMChannel> qtBadChannels = myQReport->getBadChannels();

      if (mVerbose) {
        cout << "[L1TTestsSummary:] Found QReport for ME: " << me->getTH1()->GetName() << endl;
        cout << "[L1TTestsSummary:] Result=" << qtresult << " status=" << qtstatus << " message=" << qtmessage << endl;
        cout << "[L1TTestsSummary:] Bad Channels size=" << qtBadChannels.size() << endl;
      }

      for (unsigned int i = 0; i < mProcessedLS.size() - 1; i++) {
        int binx = mL1TRateMonitor->getTH2F()->GetXaxis()->FindBin(mProcessedLS[i]);
        int biny = mL1TRateMonitor->getTH2F()->GetYaxis()->FindBin(me->getTH1()->GetName());
        mL1TRateMonitor->setBinContent(binx, biny, 100);
      }

      for (unsigned int a = 0; a < qtBadChannels.size(); a++) {
        for (unsigned int b = 0; b < mProcessedLS.size() - 1; b++) {
          // Converting bin to value
          double valueBinBad = me->getTH1()->GetBinCenter(qtBadChannels[a].getBin());

          if (valueBinBad == (mProcessedLS[b])) {
            int binx = mL1TRateMonitor->getTH2F()->GetXaxis()->FindBin(valueBinBad);
            int biny = mL1TRateMonitor->getTH2F()->GetYaxis()->FindBin(me->getTH1()->GetName());
            mL1TRateMonitor->setBinContent(binx, biny, 300);
          }
        }
      }
    }
  }

  //-> Filling the summaries
  int nBinX = mL1TRateMonitor->getTH2F()->GetXaxis()->GetNbins();
  int nBinY = mL1TRateMonitor->getTH2F()->GetYaxis()->GetNbins();
  for (int binx = 1; binx <= nBinX; binx++) {
    int GlobalStatus = 0;
    for (int biny = 2; biny <= nBinY; biny++) {
      double flag = mL1TRateMonitor->getBinContent(binx, biny);
      if (GlobalStatus < flag) {
        GlobalStatus = flag;
      }
    }

    // NOTE: Assumes mL1TSummary has same size then mL1TRateMonitor
    mL1TRateMonitor->setBinContent(binx, 1, GlobalStatus);
    mL1TSummary->setBinContent(binx, binYRate, GlobalStatus);
  }
}

//____________________________________________________________________________
// Function: updateL1TSyncMonitor
// Description:
// Note: Values certified by L1TRates are always about currentLS-1 since we
//       use rate averages from the previous LS from GT
//____________________________________________________________________________
void L1TTestsSummary::updateL1TSyncMonitor(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  igetter.setCurrentFolder(mL1TSyncPath);
  vector<string> histToMonitor = igetter.getMEs();

  for (unsigned int i = 0; i < histToMonitor.size(); i++) {
    MonitorElement *me = igetter.get(mL1TSyncPath + histToMonitor[i]);
    if (mVerbose) {
      cout << "[L1TTestsSummary:] Found ME: " << me->getTH1()->GetName() << endl;
    }

    const QReport *myQReport = me->getQReport("L1TSyncTest");  //get QReport associated to your ME
    if (myQReport) {
      float qtresult = myQReport->getQTresult();          // get QT result value
      int qtstatus = myQReport->getStatus();              // get QT status value (see table below)
      const string &qtmessage = myQReport->getMessage();  // get the whole QT result message
      vector<DQMChannel> qtBadChannels = myQReport->getBadChannels();

      if (mVerbose) {
        cout << "[L1TTestsSummary:] Found QReport for ME: " << me->getTH1()->GetName() << endl;
        cout << "[L1TTestsSummary:] Result=" << qtresult << " status=" << qtstatus << " message=" << qtmessage << endl;
        cout << "[L1TTestsSummary:] Bad Channels size=" << qtBadChannels.size() << endl;
      }

      for (unsigned int i = 0; i < mProcessedLS.size(); i++) {
        int binx = mL1TSyncMonitor->getTH2F()->GetXaxis()->FindBin(mProcessedLS[i]);
        int biny = mL1TSyncMonitor->getTH2F()->GetYaxis()->FindBin(me->getTH1()->GetName());
        mL1TSyncMonitor->setBinContent(binx, biny, 100);
      }

      for (unsigned int a = 0; a < qtBadChannels.size(); a++) {
        for (unsigned int b = 0; b < mProcessedLS.size(); b++) {
          // Converting bin to value
          double valueBinBad = me->getTH1()->GetBinCenter(qtBadChannels[a].getBin());

          if (valueBinBad == mProcessedLS[b]) {
            int binx = mL1TSyncMonitor->getTH2F()->GetXaxis()->FindBin(valueBinBad);
            int biny = mL1TSyncMonitor->getTH2F()->GetYaxis()->FindBin(me->getTH1()->GetName());
            mL1TSyncMonitor->setBinContent(binx, biny, 300);
          }
        }
      }
    }
  }

  //-> Filling the summaries
  int nBinX = mL1TSyncMonitor->getTH2F()->GetXaxis()->GetNbins();
  int nBinY = mL1TSyncMonitor->getTH2F()->GetYaxis()->GetNbins();
  for (int binx = 1; binx <= nBinX; binx++) {
    int GlobalStatus = 0;
    for (int biny = 2; biny <= nBinY; biny++) {
      double flag = mL1TSyncMonitor->getBinContent(binx, biny);
      if (GlobalStatus < flag) {
        GlobalStatus = flag;
      }
    }

    // NOTE: Assumes mL1TSummary has same size then mL1TSyncMonitor
    mL1TSyncMonitor->setBinContent(binx, 1, GlobalStatus);
    mL1TSummary->setBinContent(binx, binYSync, GlobalStatus);
  }
}

//____________________________________________________________________________
// Function: updateL1TOccupancyMonitor
// Description:
//____________________________________________________________________________
void L1TTestsSummary::updateL1TOccupancyMonitor(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  igetter.setCurrentFolder(mL1TOccupancyPath);
  vector<string> histToMonitor = igetter.getMEs();

  for (unsigned int i = 0; i < histToMonitor.size(); i++) {
    MonitorElement *me = igetter.get(mL1TOccupancyPath + histToMonitor[i]);
    if (mVerbose) {
      cout << "[L1TTestsSummary:] Found ME: " << me->getTH1()->GetName() << endl;
    }

    const QReport *myQReport = me->getQReport("L1TOccupancyTest");  //get QReport associated to your ME
    if (myQReport) {
      float qtresult = myQReport->getQTresult();          // get QT result value
      int qtstatus = myQReport->getStatus();              // get QT status value (see table below)
      const string &qtmessage = myQReport->getMessage();  // get the whole QT result message
      vector<DQMChannel> qtBadChannels = myQReport->getBadChannels();

      if (mVerbose) {
        cout << "[L1TTestsSummary:] Found QReport for ME: " << me->getTH1()->GetName() << endl;
        cout << "[L1TTestsSummary:] Result=" << qtresult << " status=" << qtstatus << " message=" << qtmessage << endl;
        cout << "[L1TTestsSummary:] Bad Channels size=" << qtBadChannels.size() << endl;
      }

      for (unsigned int i = 0; i < mProcessedLS.size(); i++) {
        int binx = mL1TOccupancyMonitor->getTH2F()->GetXaxis()->FindBin(mProcessedLS[i]);
        int biny = mL1TOccupancyMonitor->getTH2F()->GetYaxis()->FindBin(me->getTH1()->GetName());
        mL1TOccupancyMonitor->setBinContent(binx, biny, 100);
      }

      for (unsigned int a = 0; a < qtBadChannels.size(); a++) {
        for (unsigned int b = 0; b < mProcessedLS.size(); b++) {
          // Converting bin to value
          double valueBinBad = me->getTH1()->GetBinCenter(qtBadChannels[a].getBin());

          if (valueBinBad == mProcessedLS[b]) {
            int binx = mL1TOccupancyMonitor->getTH2F()->GetXaxis()->FindBin(valueBinBad);
            int biny = mL1TOccupancyMonitor->getTH2F()->GetYaxis()->FindBin(me->getTH1()->GetName());
            mL1TOccupancyMonitor->setBinContent(binx, biny, 300);
          }
        }
      }
    }
  }

  //-> Filling the summaries
  int nBinX = mL1TOccupancyMonitor->getTH2F()->GetXaxis()->GetNbins();
  int nBinY = mL1TOccupancyMonitor->getTH2F()->GetYaxis()->GetNbins();
  for (int binx = 1; binx <= nBinX; binx++) {
    int GlobalStatus = 0;
    for (int biny = 2; biny <= nBinY; biny++) {
      double flag = mL1TOccupancyMonitor->getBinContent(binx, biny);
      if (GlobalStatus < flag) {
        GlobalStatus = flag;
      }
    }

    // NOTE: Assumes mL1TSummary has same size then mL1TOccupancyMonitor
    mL1TOccupancyMonitor->setBinContent(binx, 1, GlobalStatus);
    mL1TSummary->setBinContent(binx, binYOccpancy, GlobalStatus);
  }
}

//____________________________________________________________________________
// Function: updateL1TOccupancyMonitor
// Description:
//____________________________________________________________________________
void L1TTestsSummary::updateL1TSummary(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  int nBinX = mL1TSummary->getTH2F()->GetXaxis()->GetNbins();
  for (int binx = 1; binx <= nBinX; binx++) {
    int GlobalStatus = 0;
    if (mMonitorL1TRate) {
      if (mL1TSummary->getBinContent(binx, binYRate) > GlobalStatus) {
        GlobalStatus = mL1TSummary->getBinContent(binx, binYRate);
      }
    }
    if (mMonitorL1TSync) {
      if (mL1TSummary->getBinContent(binx, binYSync) > GlobalStatus) {
        GlobalStatus = mL1TSummary->getBinContent(binx, binYSync);
      }
    }
    if (mMonitorL1TOccupancy) {
      if (mL1TSummary->getBinContent(binx, binYOccpancy) > GlobalStatus) {
        GlobalStatus = mL1TSummary->getBinContent(binx, binYOccpancy);
      }
    }
    mL1TSummary->setBinContent(binx, 1, GlobalStatus);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TTestsSummary);
