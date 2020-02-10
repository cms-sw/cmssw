/*
    Author:  Adam Hunt
    email:   ahunt@princeton.edu
*/
#include "DQM/HLXMonitor/interface/HLXMonitor.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

// STL Headers

#include <TSystem.h>
#include <cmath>
#include <iomanip>

using std::cout;
using std::endl;

HLXMonitor::HLXMonitor(const edm::ParameterSet &iConfig) {
  NUM_HLX = iConfig.getUntrackedParameter<unsigned int>("numHlx", 36);
  NUM_BUNCHES = iConfig.getUntrackedParameter<unsigned int>("numBunches", 3564);
  MAX_LS = iConfig.getUntrackedParameter<unsigned int>("maximumNumLS", 480);
  listenPort = iConfig.getUntrackedParameter<unsigned int>("SourcePort", 51001);
  OutputFilePrefix = iConfig.getUntrackedParameter<std::string>("outputFile", "lumi");
  OutputDir = iConfig.getUntrackedParameter<std::string>("outputDir", "  data");
  SavePeriod = iConfig.getUntrackedParameter<unsigned int>("SavePeriod", 10);
  NBINS = iConfig.getUntrackedParameter<unsigned int>("NBINS",
                                                      297);  // 12 BX per bin
  XMIN = iConfig.getUntrackedParameter<double>("XMIN", 0);
  XMAX = iConfig.getUntrackedParameter<double>("XMAX", 3564);
  Style = iConfig.getUntrackedParameter<std::string>("Style", "BX");
  AquireMode = iConfig.getUntrackedParameter<unsigned int>("AquireMode", 0);
  Accumulate = iConfig.getUntrackedParameter<bool>("Accumulate", true);  // all
  TriggerBX = iConfig.getUntrackedParameter<unsigned int>("TriggerBX", 50);
  MinLSBeforeSave = iConfig.getUntrackedParameter<unsigned int>("MinLSBeforeSave", 1);
  reconnTime = iConfig.getUntrackedParameter<unsigned int>("ReconnectionTime", 5);
  DistribIP1 = iConfig.getUntrackedParameter<std::string>("PrimaryHLXDAQIP", "vmepcs2f17-18");
  DistribIP2 = iConfig.getUntrackedParameter<std::string>("SecondaryHLXDAQIP", "vmepcs2f17-19");

  eventInfoFolderHLX_ = iConfig.getUntrackedParameter<std::string>("eventInfoFolderHLX", "EventInfoHLX");
  eventInfoFolder_ = iConfig.getUntrackedParameter<std::string>("eventInfoFolder", "EventInfo");
  subSystemName_ = iConfig.getUntrackedParameter<std::string>("subSystemName", "HLX");

  // Set the lumi section counter
  lsBinOld = 0;
  previousSection = 0;
  lumiSectionCount = 0;
  sectionInstantSumEt = 0;
  sectionInstantErrSumEt = 0;
  sectionInstantSumOcc1 = 0;
  sectionInstantErrSumOcc1 = 0;
  sectionInstantSumOcc2 = 0;
  sectionInstantErrSumOcc2 = 0;
  sectionInstantNorm = 0;

  // HLX Config info
  set1BelowIndex = 0;
  set1BetweenIndex = 1;
  set1AboveIndex = 2;
  set2BelowIndex = 3;
  set2BetweenIndex = 4;
  set2AboveIndex = 5;

  runNumLength = 9;
  secNumLength = 8;

  if (NUM_HLX > 36)
    NUM_HLX = 36;

  if (NUM_BUNCHES > 3564)
    NUM_BUNCHES = 3564;

  if (XMAX <= XMIN) {
    XMIN = 0;
    if (XMAX <= 0)
      XMAX = 3564;
  }

  if ((Style == "History") || (NBINS == 0)) {
    NBINS = (unsigned int)(XMAX - XMIN);
  }

  monitorName_ = iConfig.getUntrackedParameter<std::string>("monitorName", "HLX");
  // cout << "Monitor name = " << monitorName_ << endl;
  prescaleEvt_ = iConfig.getUntrackedParameter<int>("prescaleEvt", -1);
  // cout << "===>DQM event prescale = " << prescaleEvt_ << " events "<< endl;

  unsigned int HLXHFMapTemp[] = {31, 32, 33, 34, 35, 18,  // s2f07 hf-
                                 13, 14, 15, 16, 17, 0,   // s2f07 hf+
                                 25, 26, 27, 28, 29, 30,  // s2f05 hf-
                                 7,  8,  9,  10, 11, 12,  // s2f05 hf+
                                 19, 20, 21, 22, 23, 24,  // s2f02 hf-
                                 1,  2,  3,  4,  5,  6};  // s2f02 hf+

  currentRunEnded_ = true;
  runNumber_ = 0;
  expectedNibbles_ = 0;

  for (int iHLX = 0; iHLX < 36; ++iHLX) {
    HLXHFMap[iHLX] = HLXHFMapTemp[iHLX];
    // std::cout << "At " << iHLX << " Wedge " << HLXHFMap[iHLX] << std::endl;
    totalNibbles_[iHLX] = 0;
  }

  num4NibblePerLS_ = 16.0;

  connectHLXTCP();  // this was originally done in beginJob()
}

HLXMonitor::~HLXMonitor() {
  HLXTCP.Disconnect();
  EndRun();
}

// Method called once each job just before starting event loop
void HLXMonitor::connectHLXTCP() {
  HLXTCP.SetIP(DistribIP1);
  int errorCode = HLXTCP.SetPort(listenPort);
  cout << "SetPort: " << listenPort << " Success: " << errorCode << endl;
  errorCode = HLXTCP.SetMode(AquireMode);
  cout << "AquireMode: " << AquireMode << " Success: " << errorCode << endl;

  while (HLXTCP.IsConnected() == false) {
    HLXTCP.SetIP(DistribIP1);
    if (HLXTCP.Connect() != 1) {
      std::cout << "Failed to connect to " << DistribIP1 << "." << std::endl;
      sleep(1);
      std::cout << "Trying " << DistribIP2 << std::endl;
      HLXTCP.SetIP(DistribIP2);
      if (HLXTCP.Connect() == 1)
        break;
      std::cout << "Failed to connect to " << DistribIP2 << "." << std::endl;
      std::cout << " Reconnect in " << reconnTime << " seconds." << std::endl;
      sleep(reconnTime);
    }
  }
  if (HLXTCP.IsConnected() == true) {
    std::cout << "Successfully connected." << std::endl;
  }
}

// ------------ Setup the monitoring elements ---------------
void HLXMonitor::bookHistograms(DQMStore::IBooker &iBooker, edm::Run const &, edm::EventSetup const &) {
  SetupHists(iBooker);
  SetupEventInfo(iBooker);
}

void HLXMonitor::SetupHists(DQMStore::IBooker &iBooker) {
  iBooker.setCurrentFolder(monitorName_ + "/HFPlus");

  for (unsigned int iWedge = 0; iWedge < 18 && iWedge < NUM_HLX; ++iWedge) {
    std::ostringstream tempStreamer;
    tempStreamer << std::dec << std::setw(2) << std::setfill('0') << (iWedge + 1);

    std::ostringstream wedgeNum;
    wedgeNum << std::dec << (iWedge % 18) + 1;

    iBooker.setCurrentFolder(monitorName_ + "/HFPlus/Wedge" + tempStreamer.str());

    Set1Below[iWedge] =
        iBooker.book1D("Set1_Below", "HF+ Wedge " + wedgeNum.str() + ": Below Threshold 1 - Set 1", NBINS, XMIN, XMAX);
    Set1Between[iWedge] = iBooker.book1D(
        "Set1_Between", "HF+ Wedge " + wedgeNum.str() + ": Between Threshold 1 & 2 - Set 1", NBINS, XMIN, XMAX);
    Set1Above[iWedge] =
        iBooker.book1D("Set1_Above", "HF+ Wedge " + wedgeNum.str() + ": Above Threshold 2 - Set 1", NBINS, XMIN, XMAX);
    Set2Below[iWedge] =
        iBooker.book1D("Set2_Below", "HF+ Wedge " + wedgeNum.str() + ": Below Threshold 1 - Set 2", NBINS, XMIN, XMAX);
    Set2Between[iWedge] = iBooker.book1D(
        "Set2_Between", "HF+ Wedge " + wedgeNum.str() + ": Between Threshold 1 & 2 - Set 2", NBINS, XMIN, XMAX);
    Set2Above[iWedge] =
        iBooker.book1D("Set2_Above", "HF+ Wedge " + wedgeNum.str() + ": Above Threshold 2 - Set 2", NBINS, XMIN, XMAX);
    ETSum[iWedge] = iBooker.book1D("ETSum", "HF+ Wedge " + wedgeNum.str() + ": Transverse Energy", NBINS, XMIN, XMAX);
  }

  if (NUM_HLX > 17) {
    iBooker.setCurrentFolder(monitorName_ + "/HFMinus");

    for (unsigned int iWedge = 18; iWedge < NUM_HLX; ++iWedge) {
      std::ostringstream tempStreamer;
      tempStreamer << std::dec << std::setw(2) << std::setfill('0') << (iWedge + 1);

      std::ostringstream wedgeNum;
      wedgeNum << std::dec << (iWedge % 18) + 1;

      iBooker.setCurrentFolder(monitorName_ + "/HFMinus/Wedge" + tempStreamer.str());
      Set1Below[iWedge] = iBooker.book1D(
          "Set1_Below", "HF- Wedge " + wedgeNum.str() + ": Below Threshold 1 - Set 1", NBINS, XMIN, XMAX);
      Set1Between[iWedge] = iBooker.book1D(
          "Set1_Between", "HF- Wedge " + wedgeNum.str() + ": Between Threshold 1 & 2 - Set 1", NBINS, XMIN, XMAX);
      Set1Above[iWedge] = iBooker.book1D(
          "Set1_Above", "HF- Wedge " + wedgeNum.str() + ": Above Threshold 2 - Set 1", NBINS, XMIN, XMAX);
      Set2Below[iWedge] = iBooker.book1D(
          "Set2_Below", "HF- Wedge " + wedgeNum.str() + ": Below Threshold 1 - Set 2", NBINS, XMIN, XMAX);
      Set2Between[iWedge] = iBooker.book1D(
          "Set2_Between", "HF- Wedge " + wedgeNum.str() + ": Between Threshold 1 & 2 - Set 2", NBINS, XMIN, XMAX);
      Set2Above[iWedge] = iBooker.book1D(
          "Set2_Above", "HF- Wedge " + wedgeNum.str() + ": Above Threshold 2 - Set 2", NBINS, XMIN, XMAX);
      ETSum[iWedge] = iBooker.book1D("ETSum", "HF- Wedge " + wedgeNum.str() + ": Transverse Energy", NBINS, XMIN, XMAX);
    }
  }

  if (Style == "BX") {
    OccXAxisTitle = "Bunch Crossing";
    OccYAxisTitle = "Tower Occupancy";
    EtXAxisTitle = "Bunch Crossing";
    EtYAxisTitle = "E_{T} Sum";
  } else if (Style == "Distribution") {
    OccXAxisTitle = "Tower Occupancy";
    OccYAxisTitle = "Count";
    EtXAxisTitle = "E_{T} Sum";
    EtYAxisTitle = "Count";
  }

  for (unsigned int iWedge = 0; iWedge < NUM_HLX; ++iWedge) {
    Set1Below[iWedge]->setAxisTitle(OccXAxisTitle, 1);
    Set1Below[iWedge]->setAxisTitle(OccYAxisTitle, 2);
    Set1Between[iWedge]->setAxisTitle(OccXAxisTitle, 1);
    Set1Between[iWedge]->setAxisTitle(OccYAxisTitle, 2);
    Set1Above[iWedge]->setAxisTitle(OccXAxisTitle, 1);
    Set1Above[iWedge]->setAxisTitle(OccYAxisTitle, 2);
    Set2Below[iWedge]->setAxisTitle(OccXAxisTitle, 1);
    Set2Below[iWedge]->setAxisTitle(OccYAxisTitle, 2);
    Set2Between[iWedge]->setAxisTitle(OccXAxisTitle, 1);
    Set2Between[iWedge]->setAxisTitle(OccYAxisTitle, 2);
    Set2Above[iWedge]->setAxisTitle(OccXAxisTitle, 1);
    Set2Above[iWedge]->setAxisTitle(OccYAxisTitle, 2);
    ETSum[iWedge]->setAxisTitle(EtXAxisTitle, 1);
    ETSum[iWedge]->setAxisTitle(EtYAxisTitle, 2);
  }

  // Comparison Histograms

  iBooker.setCurrentFolder(monitorName_ + "/HFCompare");

  std::string CompXTitle = "HF Wedge";
  std::string CompEtSumYTitle = "E_{T} Sum per active tower";
  std::string CompOccYTitle = "Occupancy per active tower";

  HFCompareEtSum = iBooker.book1D("HFCompareEtSum", "E_{T} Sum", NUM_HLX, 0, NUM_HLX);
  HFCompareEtSum->setAxisTitle(CompXTitle, 1);
  HFCompareEtSum->setAxisTitle(CompEtSumYTitle, 2);

  HFCompareOccBelowSet1 =
      iBooker.book1D("HFCompareOccBelowSet1", "Occupancy Below Threshold 1 - Set 1", NUM_HLX, 0, NUM_HLX);
  HFCompareOccBelowSet1->setAxisTitle(CompXTitle, 1);
  HFCompareOccBelowSet1->setAxisTitle(CompOccYTitle, 2);

  HFCompareOccBetweenSet1 =
      iBooker.book1D("HFCompareOccBetweenSet1", "Occupancy Between Threshold 1 & 2 - Set 1", NUM_HLX, 0, NUM_HLX);
  HFCompareOccBetweenSet1->setAxisTitle(CompXTitle, 1);
  HFCompareOccBetweenSet1->setAxisTitle(CompOccYTitle, 2);

  HFCompareOccAboveSet1 =
      iBooker.book1D("HFCompareOccAboveSet1", "Occupancy Above Threshold 2 - Set 1", NUM_HLX, 0, NUM_HLX);
  HFCompareOccAboveSet1->setAxisTitle(CompXTitle, 1);
  HFCompareOccAboveSet1->setAxisTitle(CompOccYTitle, 2);

  HFCompareOccBelowSet2 =
      iBooker.book1D("HFCompareOccBelowSet2", "Occupancy Below Threshold 1 - Set 2", NUM_HLX, 0, NUM_HLX);
  HFCompareOccBelowSet2->setAxisTitle(CompXTitle, 1);
  HFCompareOccBelowSet2->setAxisTitle(CompOccYTitle, 2);

  HFCompareOccBetweenSet2 =
      iBooker.book1D("HFCompareOccBetweenSet2", "Occupancy Between Threshold 1 & 2 - Set 2", NUM_HLX, 0, NUM_HLX);
  HFCompareOccBetweenSet2->setAxisTitle(CompXTitle, 1);
  HFCompareOccBetweenSet2->setAxisTitle(CompOccYTitle, 2);

  HFCompareOccAboveSet2 =
      iBooker.book1D("HFCompareOccAboveSet2", "Occupancy Above Threshold 2 - Set 2", NUM_HLX, 0, NUM_HLX);
  HFCompareOccAboveSet2->setAxisTitle(CompXTitle, 1);
  HFCompareOccAboveSet2->setAxisTitle(CompOccYTitle, 2);

  // Average Histograms

  iBooker.setCurrentFolder(monitorName_ + "/Average");

  int OccBins = 10000;  // This does absolutely nothing.
  double OccMin = 0;
  double OccMax = 0;  // If min and max are zero, no bounds on the data are set.

  int EtSumBins = 10000;  // This does absolutely nothing.  The Variable is not
                          // used in the function.
  double EtSumMin = 0;
  double EtSumMax = 0;  // If min and max are zero, no bounds on the data are set.

  std::string errorOpt = "i";

  std::string AvgXTitle = "HF Wedge";
  std::string AvgEtSumYTitle = "Average E_{T} Sum";
  std::string AvgOccYTitle = "Average Tower Occupancy";

  AvgEtSum = iBooker.bookProfile("AvgEtSum", "Average E_{T} Sum", NUM_HLX, 0, NUM_HLX, EtSumBins, EtSumMin, EtSumMax);
  AvgEtSum->setAxisTitle(AvgXTitle, 1);
  AvgEtSum->setAxisTitle(AvgEtSumYTitle, 2);

  AvgOccBelowSet1 = iBooker.bookProfile("AvgOccBelowSet1",
                                        "Average Occupancy Below Threshold 1 - Set1",
                                        NUM_HLX,
                                        0,
                                        NUM_HLX,
                                        OccBins,
                                        OccMin,
                                        OccMax,
                                        errorOpt.c_str());
  AvgOccBelowSet1->setAxisTitle(AvgXTitle, 1);
  AvgOccBelowSet1->setAxisTitle(AvgOccYTitle, 2);

  AvgOccBetweenSet1 = iBooker.bookProfile("AvgOccBetweenSet1",
                                          "Average Occupancy Between Threhold 1 & 2 - Set1",
                                          NUM_HLX,
                                          0,
                                          NUM_HLX,
                                          OccBins,
                                          OccMin,
                                          OccMax,
                                          errorOpt.c_str());
  AvgOccBetweenSet1->setAxisTitle(AvgXTitle, 1);
  AvgOccBetweenSet1->setAxisTitle(AvgOccYTitle, 2);

  AvgOccAboveSet1 = iBooker.bookProfile("AvgOccAboveSet1",
                                        "Average Occupancy Above Threshold 2 - Set1",
                                        NUM_HLX,
                                        0,
                                        NUM_HLX,
                                        OccBins,
                                        OccMin,
                                        OccMax,
                                        errorOpt.c_str());
  AvgOccAboveSet1->setAxisTitle(AvgXTitle, 1);
  AvgOccAboveSet1->setAxisTitle(AvgOccYTitle, 2);

  AvgOccBelowSet2 = iBooker.bookProfile("AvgOccBelowSet2",
                                        "Average Occupancy Below Threshold 1 - Set2",
                                        NUM_HLX,
                                        0,
                                        NUM_HLX,
                                        OccBins,
                                        OccMin,
                                        OccMax,
                                        errorOpt.c_str());
  AvgOccBelowSet2->setAxisTitle(AvgXTitle, 1);
  AvgOccBelowSet2->setAxisTitle(AvgOccYTitle, 2);

  AvgOccBetweenSet2 = iBooker.bookProfile("AvgOccBetweenSet2",
                                          "Average Occupancy Between Threshold 1 & 2 - Set2",
                                          NUM_HLX,
                                          0,
                                          NUM_HLX,
                                          OccBins,
                                          OccMin,
                                          OccMax,
                                          errorOpt.c_str());
  AvgOccBetweenSet2->setAxisTitle(AvgXTitle, 1);
  AvgOccBetweenSet2->setAxisTitle(AvgOccYTitle, 2);

  AvgOccAboveSet2 = iBooker.bookProfile("AvgOccAboveSet2",
                                        "Average Occupancy Above Threshold 2 - Set2",
                                        NUM_HLX,
                                        0,
                                        NUM_HLX,
                                        OccBins,
                                        OccMin,
                                        OccMax,
                                        errorOpt.c_str());
  AvgOccAboveSet2->setAxisTitle(AvgXTitle, 1);
  AvgOccAboveSet2->setAxisTitle(AvgOccYTitle, 2);

  // Luminosity Histograms
  iBooker.setCurrentFolder(monitorName_ + "/Luminosity");

  std::string LumiXTitle = "Bunch Crossing";
  std::string LumiEtSumYTitle = "Luminosity: E_{T} Sum";
  std::string LumiOccYTitle = "Luminosity: Occupancy";

  LumiAvgEtSum = iBooker.bookProfile(
      "LumiAvgEtSum", "Average Luminosity ", int(XMAX - XMIN), XMIN, XMAX, EtSumBins, EtSumMin, EtSumMax);
  LumiAvgEtSum->setAxisTitle(LumiXTitle, 1);
  LumiAvgEtSum->setAxisTitle(LumiEtSumYTitle, 2);

  LumiAvgOccSet1 = iBooker.bookProfile(
      "LumiAvgOccSet1", "Average Luminosity - Set 1", int(XMAX - XMIN), XMIN, XMAX, OccBins, OccMax, OccMin);
  LumiAvgOccSet1->setAxisTitle(LumiXTitle, 1);
  LumiAvgOccSet1->setAxisTitle(LumiOccYTitle, 2);

  LumiAvgOccSet2 = iBooker.bookProfile(
      "LumiAvgOccSet2", "Average Luminosity - Set 2", int(XMAX - XMIN), XMIN, XMAX, OccBins, OccMax, OccMin);
  LumiAvgOccSet2->setAxisTitle(LumiXTitle, 1);
  LumiAvgOccSet2->setAxisTitle(LumiOccYTitle, 2);

  LumiInstantEtSum = iBooker.book1D("LumiInstantEtSum", "Instantaneous Luminosity ", int(XMAX - XMIN), XMIN, XMAX);
  LumiInstantEtSum->setAxisTitle(LumiXTitle, 1);
  LumiInstantEtSum->setAxisTitle(LumiEtSumYTitle, 2);

  LumiInstantOccSet1 =
      iBooker.book1D("LumiInstantOccSet1", "Instantaneous Luminosity - Set 1", int(XMAX - XMIN), XMIN, XMAX);
  LumiInstantOccSet1->setAxisTitle(LumiXTitle, 1);
  LumiInstantOccSet1->setAxisTitle(LumiOccYTitle, 2);

  LumiInstantOccSet2 =
      iBooker.book1D("LumiInstantOccSet2", "Instantaneous Luminosity - Set 2", int(XMAX - XMIN), XMIN, XMAX);
  LumiInstantOccSet2->setAxisTitle(LumiXTitle, 1);
  LumiInstantOccSet2->setAxisTitle(LumiOccYTitle, 2);

  LumiIntegratedEtSum = iBooker.book1D("LumiIntegratedEtSum", "Integrated Luminosity ", int(XMAX - XMIN), XMIN, XMAX);
  LumiIntegratedEtSum->setAxisTitle(LumiXTitle, 1);
  LumiIntegratedEtSum->setAxisTitle(LumiEtSumYTitle, 2);

  LumiIntegratedOccSet1 =
      iBooker.book1D("LumiIntegratedOccSet1", "Integrated Luminosity - Set 1", int(XMAX - XMIN), XMIN, XMAX);
  LumiIntegratedOccSet1->setAxisTitle(LumiXTitle, 1);
  LumiIntegratedOccSet1->setAxisTitle(LumiOccYTitle, 2);

  LumiIntegratedOccSet2 =
      iBooker.book1D("LumiIntegratedOccSet2", "Integrated Luminosity - Set 2", int(XMAX - XMIN), XMIN, XMAX);
  LumiIntegratedOccSet2->setAxisTitle(LumiXTitle, 1);
  LumiIntegratedOccSet2->setAxisTitle(LumiOccYTitle, 2);

  // Sanity check sum histograms
  iBooker.setCurrentFolder(monitorName_ + "/CheckSums");

  std::string sumXTitle = "HF Wedge";
  std::string sumYTitle = "Occupancy Sum (Below+Above+Between)";

  SumAllOccSet1 =
      iBooker.bookProfile("SumAllOccSet1", "Occupancy Check - Set 1", NUM_HLX, 0, NUM_HLX, OccBins, OccMax, OccMin);
  SumAllOccSet1->setAxisTitle(sumXTitle, 1);
  SumAllOccSet1->setAxisTitle(sumYTitle, 2);

  SumAllOccSet2 =
      iBooker.bookProfile("SumAllOccSet2", "Occupancy Check - Set 2", NUM_HLX, 0, NUM_HLX, OccBins, OccMax, OccMin);
  SumAllOccSet2->setAxisTitle(sumXTitle, 1);
  SumAllOccSet2->setAxisTitle(sumYTitle, 2);

  MissingDQMDataCheck = iBooker.book1D("MissingDQMDataCheck", "Missing Data Count", 1, 0, 1);
  MissingDQMDataCheck->setAxisTitle("", 1);
  MissingDQMDataCheck->setAxisTitle("Number Missing Nibbles", 2);

  // Signal & Background monitoring histograms
  iBooker.setCurrentFolder(monitorName_ + "/SigBkgLevels");

  MaxInstLumiBX1 = iBooker.book1D("MaxInstLumiBX1", "Max Instantaneous Luminosity BX: 1st", 10000, -1e-5, 0.01);
  MaxInstLumiBX1->setAxisTitle("Max Inst. L (10^{30}cm^{-2}s^{-1})", 1);
  MaxInstLumiBX1->setAxisTitle("Entries", 2);
  MaxInstLumiBX2 = iBooker.book1D("MaxInstLumiBX2", "Max Instantaneous Luminosity BX: 2nd", 10000, -1e-5, 0.01);
  MaxInstLumiBX2->setAxisTitle("Max Inst. L (10^{30}cm^{-2}s^{-1})", 1);
  MaxInstLumiBX2->setAxisTitle("Entries", 2);
  MaxInstLumiBX3 = iBooker.book1D("MaxInstLumiBX3", "Max Instantaneous Luminosity BX: 3rd", 10000, -1e-5, 0.01);
  MaxInstLumiBX3->setAxisTitle("Max Inst. L (10^{30}cm^{-2}s^{-1})", 1);
  MaxInstLumiBX3->setAxisTitle("Entries", 2);
  MaxInstLumiBX4 = iBooker.book1D("MaxInstLumiBX4", "Max Instantaneous Luminosity BX: 4th", 10000, -1e-5, 0.01);
  MaxInstLumiBX4->setAxisTitle("Max Inst. L (10^{30}cm^{-2}s^{-1})", 1);
  MaxInstLumiBX4->setAxisTitle("Entries", 2);

  MaxInstLumiBXNum1 = iBooker.book1D("MaxInstLumiBXNum1", "BX Number of Max: 1st", 3564, 0, 3564);
  MaxInstLumiBXNum1->setAxisTitle("BX", 1);
  MaxInstLumiBXNum1->setAxisTitle("Num Time Max", 2);
  MaxInstLumiBXNum2 = iBooker.book1D("MaxInstLumiBXNum2", "BX Number of Max: 2nd", 3564, 0, 3564);
  MaxInstLumiBXNum2->setAxisTitle("BX", 1);
  MaxInstLumiBXNum2->setAxisTitle("Num Time Max", 2);
  MaxInstLumiBXNum3 = iBooker.book1D("MaxInstLumiBXNum3", "BX Number of Max: 3rd", 3564, 0, 3564);
  MaxInstLumiBXNum3->setAxisTitle("BX", 1);
  MaxInstLumiBXNum3->setAxisTitle("Num Time Max", 2);
  MaxInstLumiBXNum4 = iBooker.book1D("MaxInstLumiBXNum4", "BX Number of Max: 4th", 3564, 0, 3564);
  MaxInstLumiBXNum4->setAxisTitle("BX", 1);
  MaxInstLumiBXNum4->setAxisTitle("Num Time Max", 2);

  // History histograms
  iBooker.setCurrentFolder(monitorName_ + "/HistoryRaw");

  std::string HistXTitle = "Time (LS)";
  std::string RecentHistXTitle = "Time (LS/16)";
  std::string HistEtSumYTitle = "Average E_{T} Sum";
  std::string HistOccYTitle = "Average Occupancy";
  std::string HistLumiYTitle = "Luminosity";
  std::string HistLumiErrorYTitle = "Luminosity Error (%)";
  std::string BXvsTimeXTitle = "Time (LS)";
  std::string BXvsTimeYTitle = "BX";

  // Et Sum histories
  HistAvgEtSumHFP = iBooker.bookProfile(
      "HistAvgEtSumHFP", "Average Et Sum: HF+", MAX_LS, 0.5, (double)MAX_LS + 0.5, EtSumBins, EtSumMin, EtSumMax);
  HistAvgEtSumHFP->setAxisTitle(HistXTitle, 1);
  HistAvgEtSumHFP->setAxisTitle(HistEtSumYTitle, 2);

  HistAvgEtSumHFM = iBooker.bookProfile(
      "HistAvgEtSumHFM", "Average Et Sum: HF-", MAX_LS, 0.5, (double)MAX_LS + 0.5, EtSumBins, EtSumMin, EtSumMax);
  HistAvgEtSumHFM->setAxisTitle(HistXTitle, 1);
  HistAvgEtSumHFM->setAxisTitle(HistEtSumYTitle, 2);

  // Tower Occupancy Histories
  HistAvgOccBelowSet1HFP = iBooker.bookProfile("HistAvgOccBelowSet1HFP",
                                               "Average Occ Set1Below: HF+",
                                               MAX_LS,
                                               0.5,
                                               (double)MAX_LS + 0.5,
                                               OccBins,
                                               OccMin,
                                               OccMax);
  HistAvgOccBelowSet1HFP->setAxisTitle(HistXTitle, 1);
  HistAvgOccBelowSet1HFP->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccBelowSet1HFM = iBooker.bookProfile("HistAvgOccBelowSet1HFM",
                                               "Average Occ Set1Below: HF-",
                                               MAX_LS,
                                               0.5,
                                               (double)MAX_LS + 0.5,
                                               OccBins,
                                               OccMin,
                                               OccMax);
  HistAvgOccBelowSet1HFM->setAxisTitle(HistXTitle, 1);
  HistAvgOccBelowSet1HFM->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccBetweenSet1HFP = iBooker.bookProfile("HistAvgOccBetweenSet1HFP",
                                                 "Average Occ Set1Between: HF+",
                                                 MAX_LS,
                                                 0.5,
                                                 (double)MAX_LS + 0.5,
                                                 OccBins,
                                                 OccMin,
                                                 OccMax);
  HistAvgOccBetweenSet1HFP->setAxisTitle(HistXTitle, 1);
  HistAvgOccBetweenSet1HFP->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccBetweenSet1HFM = iBooker.bookProfile("HistAvgOccBetweenSet1HFM",
                                                 "Average Occ Set1Between: HF-",
                                                 MAX_LS,
                                                 0.5,
                                                 (double)MAX_LS + 0.5,
                                                 OccBins,
                                                 OccMin,
                                                 OccMax);
  HistAvgOccBetweenSet1HFM->setAxisTitle(HistXTitle, 1);
  HistAvgOccBetweenSet1HFM->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccAboveSet1HFP = iBooker.bookProfile("HistAvgOccAboveSet1HFP",
                                               "Average Occ Set1Above: HF+",
                                               MAX_LS,
                                               0.5,
                                               (double)MAX_LS + 0.5,
                                               OccBins,
                                               OccMin,
                                               OccMax);
  HistAvgOccAboveSet1HFP->setAxisTitle(HistXTitle, 1);
  HistAvgOccAboveSet1HFP->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccAboveSet1HFM = iBooker.bookProfile("HistAvgOccAboveSet1HFM",
                                               "Average Occ Set1Above: HF-",
                                               MAX_LS,
                                               0.5,
                                               (double)MAX_LS + 0.5,
                                               OccBins,
                                               OccMin,
                                               OccMax);
  HistAvgOccAboveSet1HFM->setAxisTitle(HistXTitle, 1);
  HistAvgOccAboveSet1HFM->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccBelowSet2HFP = iBooker.bookProfile("HistAvgOccBelowSet2HFP",
                                               "Average Occ Set2Below: HF+",
                                               MAX_LS,
                                               0.5,
                                               (double)MAX_LS + 0.5,
                                               OccBins,
                                               OccMin,
                                               OccMax);
  HistAvgOccBelowSet2HFP->setAxisTitle(HistXTitle, 1);
  HistAvgOccBelowSet2HFP->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccBelowSet2HFM = iBooker.bookProfile("HistAvgOccBelowSet2HFM",
                                               "Average Occ Set2Below: HF-",
                                               MAX_LS,
                                               0.5,
                                               (double)MAX_LS + 0.5,
                                               OccBins,
                                               OccMin,
                                               OccMax);
  HistAvgOccBelowSet2HFM->setAxisTitle(HistXTitle, 1);
  HistAvgOccBelowSet2HFM->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccBetweenSet2HFP = iBooker.bookProfile("HistAvgOccBetweenSet2HFP",
                                                 "Average Occ Set2Between: HF+",
                                                 MAX_LS,
                                                 0.5,
                                                 (double)MAX_LS + 0.5,
                                                 OccBins,
                                                 OccMin,
                                                 OccMax);
  HistAvgOccBetweenSet2HFP->setAxisTitle(HistXTitle, 1);
  HistAvgOccBetweenSet2HFP->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccBetweenSet2HFM = iBooker.bookProfile("HistAvgOccBetweenSet2HFM",
                                                 "Average Occ Set2Between: HF-",
                                                 MAX_LS,
                                                 0.5,
                                                 (double)MAX_LS + 0.5,
                                                 OccBins,
                                                 OccMin,
                                                 OccMax);
  HistAvgOccBetweenSet2HFM->setAxisTitle(HistXTitle, 1);
  HistAvgOccBetweenSet2HFM->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccAboveSet2HFP = iBooker.bookProfile("HistAvgOccAboveSet2HFP",
                                               "Average Occ Set2Above: HF+",
                                               MAX_LS,
                                               0.5,
                                               (double)MAX_LS + 0.5,
                                               OccBins,
                                               OccMin,
                                               OccMax);
  HistAvgOccAboveSet2HFP->setAxisTitle(HistXTitle, 1);
  HistAvgOccAboveSet2HFP->setAxisTitle(HistOccYTitle, 2);

  HistAvgOccAboveSet2HFM = iBooker.bookProfile("HistAvgOccAboveSet2HFM",
                                               "Average Occ Set2Above: HF-",
                                               MAX_LS,
                                               0.5,
                                               (double)MAX_LS + 0.5,
                                               OccBins,
                                               OccMin,
                                               OccMax);
  HistAvgOccAboveSet2HFM->setAxisTitle(HistXTitle, 1);
  HistAvgOccAboveSet2HFM->setAxisTitle(HistOccYTitle, 2);

  // Et Sum histories
  BXvsTimeAvgEtSumHFP = iBooker.book2D("BXvsTimeAvgEtSumHFP",
                                       "Average Et Sum: HF+",
                                       MAX_LS,
                                       0.5,
                                       (double)MAX_LS + 0.5,
                                       NBINS,
                                       (double)XMIN,
                                       (double)XMAX);
  BXvsTimeAvgEtSumHFP->setAxisTitle(BXvsTimeXTitle, 1);
  BXvsTimeAvgEtSumHFP->setAxisTitle(BXvsTimeYTitle, 2);

  BXvsTimeAvgEtSumHFM = iBooker.book2D("BXvsTimeAvgEtSumHFM",
                                       "Average Et Sum: HF-",
                                       MAX_LS,
                                       0.5,
                                       (double)MAX_LS + 0.5,
                                       NBINS,
                                       (double)XMIN,
                                       (double)XMAX);
  BXvsTimeAvgEtSumHFM->setAxisTitle(BXvsTimeXTitle, 1);
  BXvsTimeAvgEtSumHFM->setAxisTitle(BXvsTimeYTitle, 2);

  iBooker.setCurrentFolder(monitorName_ + "/HistoryLumi");

  // Lumi Histories
  HistAvgLumiEtSum = iBooker.bookProfile("HistAvgLumiEtSum",
                                         "Average Instant Luminosity: Et Sum",
                                         MAX_LS,
                                         0.5,
                                         (double)MAX_LS + 0.5,
                                         EtSumBins,
                                         EtSumMin,
                                         EtSumMax);
  HistAvgLumiEtSum->setAxisTitle(HistXTitle, 1);
  HistAvgLumiEtSum->setAxisTitle(HistLumiYTitle, 2);

  HistAvgLumiOccSet1 = iBooker.bookProfile("HistAvgLumiOccSet1",
                                           "Average Instant Luminosity: Occ Set1",
                                           MAX_LS,
                                           0.5,
                                           (double)MAX_LS + 0.5,
                                           OccBins,
                                           OccMin,
                                           OccMax);
  HistAvgLumiOccSet1->setAxisTitle(HistXTitle, 1);
  HistAvgLumiOccSet1->setAxisTitle(HistLumiYTitle, 2);

  HistAvgLumiOccSet2 = iBooker.bookProfile("HistAvgLumiOccSet2",
                                           "Average Instant Luminosity: Occ Set2",
                                           MAX_LS,
                                           0.5,
                                           (double)MAX_LS + 0.5,
                                           OccBins,
                                           OccMin,
                                           OccMax);
  HistAvgLumiOccSet2->setAxisTitle(HistXTitle, 1);
  HistAvgLumiOccSet2->setAxisTitle(HistLumiYTitle, 2);

  HistInstantLumiEtSum =
      iBooker.book1D("HistInstantLumiEtSum", "Instant Luminosity: Et Sum", MAX_LS, 0.5, (double)MAX_LS + 0.5);
  HistInstantLumiEtSum->setAxisTitle(HistXTitle, 1);
  HistInstantLumiEtSum->setAxisTitle(HistLumiYTitle, 2);

  HistInstantLumiOccSet1 =
      iBooker.book1D("HistInstantLumiOccSet1", "Instant Luminosity: Occ Set1", MAX_LS, 0.5, (double)MAX_LS + 0.5);
  HistInstantLumiOccSet1->setAxisTitle(HistXTitle, 1);
  HistInstantLumiOccSet1->setAxisTitle(HistLumiYTitle, 2);

  HistInstantLumiOccSet2 =
      iBooker.book1D("HistInstantLumiOccSet2", "Instant Luminosity: Occ Set2", MAX_LS, 0.5, (double)MAX_LS + 0.5);
  HistInstantLumiOccSet2->setAxisTitle(HistXTitle, 1);
  HistInstantLumiOccSet2->setAxisTitle(HistLumiYTitle, 2);

  HistInstantLumiEtSumError =
      iBooker.book1D("HistInstantLumiEtSumError", "Luminosity Error: Et Sum", MAX_LS, 0.5, (double)MAX_LS + 0.5);
  HistInstantLumiEtSumError->setAxisTitle(HistXTitle, 1);
  HistInstantLumiEtSumError->setAxisTitle(HistLumiErrorYTitle, 2);

  HistInstantLumiOccSet1Error =
      iBooker.book1D("HistInstantLumiOccSet1Error", "Luminosity Error: Occ Set1", MAX_LS, 0.5, (double)MAX_LS + 0.5);
  HistInstantLumiOccSet1Error->setAxisTitle(HistXTitle, 1);
  HistInstantLumiOccSet1Error->setAxisTitle(HistLumiErrorYTitle, 2);

  HistInstantLumiOccSet2Error =
      iBooker.book1D("HistInstantLumiOccSet2Error", "Luminosity Error: Occ Set2", MAX_LS, 0.5, (double)MAX_LS + 0.5);
  HistInstantLumiOccSet2Error->setAxisTitle(HistXTitle, 1);
  HistInstantLumiOccSet2Error->setAxisTitle(HistLumiErrorYTitle, 2);

  HistIntegratedLumiEtSum =
      iBooker.book1D("HistIntegratedLumiEtSum", "Integrated Luminosity: Et Sum", MAX_LS, 0.5, (double)MAX_LS + 0.5);
  HistIntegratedLumiEtSum->setAxisTitle(HistXTitle, 1);
  HistIntegratedLumiEtSum->setAxisTitle(HistLumiYTitle, 2);

  HistIntegratedLumiOccSet1 =
      iBooker.book1D("HistIntegratedLumiOccSet1", "Integrated Luminosity: Occ Set1", MAX_LS, 0.5, (double)MAX_LS + 0.5);
  HistIntegratedLumiOccSet1->setAxisTitle(HistXTitle, 1);
  HistIntegratedLumiOccSet1->setAxisTitle(HistLumiYTitle, 2);

  HistIntegratedLumiOccSet2 =
      iBooker.book1D("HistIntegratedLumiOccSet2", "Integrated Luminosity: Occ Set2", MAX_LS, 0.5, (double)MAX_LS + 0.5);
  HistIntegratedLumiOccSet2->setAxisTitle(HistXTitle, 1);
  HistIntegratedLumiOccSet2->setAxisTitle(HistLumiYTitle, 2);

  iBooker.setCurrentFolder(monitorName_ + "/RecentHistoryLumi");

  // Lumi Recent Histories (past 128 short sections)
  RecentInstantLumiEtSum =
      iBooker.book1D("RecentInstantLumiEtSum", "Instant Luminosity: Et Sum", 128, 0.5, (double)128 + 0.5);
  RecentInstantLumiEtSum->setAxisTitle(RecentHistXTitle, 1);
  RecentInstantLumiEtSum->setAxisTitle(HistLumiYTitle, 2);

  RecentInstantLumiOccSet1 =
      iBooker.book1D("RecentInstantLumiOccSet1", "Instant Luminosity: Occ Set1", 128, 0.5, (double)128 + 0.5);
  RecentInstantLumiOccSet1->setAxisTitle(RecentHistXTitle, 1);
  RecentInstantLumiOccSet1->setAxisTitle(HistLumiYTitle, 2);

  RecentInstantLumiOccSet2 =
      iBooker.book1D("RecentInstantLumiOccSet2", "Instant Luminosity: Occ Set2", 128, 0.5, (double)128 + 0.5);
  RecentInstantLumiOccSet2->setAxisTitle(RecentHistXTitle, 1);
  RecentInstantLumiOccSet2->setAxisTitle(HistLumiYTitle, 2);

  RecentIntegratedLumiEtSum =
      iBooker.book1D("RecentIntegratedLumiEtSum", "Integrated Luminosity: Et Sum", 128, 0.5, (double)128 + 0.5);
  RecentIntegratedLumiEtSum->setAxisTitle(RecentHistXTitle, 1);
  RecentIntegratedLumiEtSum->setAxisTitle(HistLumiYTitle, 2);

  RecentIntegratedLumiOccSet1 =
      iBooker.book1D("RecentIntegratedLumiOccSet1", "Integrated Luminosity: Occ Set1", 128, 0.5, (double)128 + 0.5);
  RecentIntegratedLumiOccSet1->setAxisTitle(RecentHistXTitle, 1);
  RecentIntegratedLumiOccSet1->setAxisTitle(HistLumiYTitle, 2);

  RecentIntegratedLumiOccSet2 =
      iBooker.book1D("RecentIntegratedLumiOccSet2", "Integrated Luminosity: Occ Set2", 128, 0.5, (double)128 + 0.5);
  RecentIntegratedLumiOccSet2->setAxisTitle(RecentHistXTitle, 1);
  RecentIntegratedLumiOccSet2->setAxisTitle(HistLumiYTitle, 2);
}

void HLXMonitor::SetupEventInfo(DQMStore::IBooker &iBooker) {
  using std::string;

  string currentfolder = subSystemName_ + "/" + eventInfoFolderHLX_;
  // cout << "currentfolder " << currentfolder << endl;

  iBooker.setCurrentFolder(currentfolder);

  pEvent_ = 0;
  evtRateCount_ = 0;
  gettimeofday(&currentTime_, nullptr);
  lastAvgTime_ = currentTime_;
  evtRateWindow_ = 0.5;

  // Event specific contents
  runId_ = iBooker.bookInt("iRun");
  lumisecId_ = iBooker.bookInt("iLumiSection");

  eventId_ = iBooker.bookInt("iEvent");
  eventId_->Fill(-1);
  eventTimeStamp_ = iBooker.bookFloat("eventTimeStamp");

  iBooker.setCurrentFolder(currentfolder);
  // Process specific contents
  processTimeStamp_ = iBooker.bookFloat("processTimeStamp");
  processTimeStamp_->Fill(getUTCtime(&currentTime_));
  processLatency_ = iBooker.bookFloat("processLatency");
  processTimeStamp_->Fill(-1);
  processEvents_ = iBooker.bookInt("processedEvents");
  processEvents_->Fill(pEvent_);
  processEventRate_ = iBooker.bookFloat("processEventRate");
  processEventRate_->Fill(-1);
  nUpdates_ = iBooker.bookInt("processUpdates");
  nUpdates_->Fill(-1);

  // Static Contents
  processId_ = iBooker.bookInt("processID");
  processId_->Fill(gSystem->GetPid());
  processStartTimeStamp_ = iBooker.bookFloat("processStartTimeStamp");
  processStartTimeStamp_->Fill(getUTCtime(&currentTime_));
  runStartTimeStamp_ = iBooker.bookFloat("runStartTimeStamp");
  hostName_ = iBooker.bookString("hostName", gSystem->HostName());
  processName_ = iBooker.bookString("processName", subSystemName_);
  workingDir_ = iBooker.bookString("workingDir", gSystem->pwd());
  cmsswVer_ = iBooker.bookString("CMSSW_Version", edm::getReleaseVersion());

  // Go to the standard EventInfo folder (in the case online case where this
  // is different).
  currentfolder = subSystemName_ + "/" + eventInfoFolder_;
  iBooker.setCurrentFolder(currentfolder);

  reportSummary_ = iBooker.bookFloat("reportSummary");
  reportSummaryMap_ = iBooker.book2D("reportSummaryMap", "reportSummaryMap", 18, 0., 18., 2, -1.5, 1.5);

  currentfolder = subSystemName_ + "/" + eventInfoFolderHLX_;
  iBooker.setCurrentFolder(currentfolder);

  TH2F *summaryHist = reportSummaryMap_->getTH2F();
  summaryHist->GetYaxis()->SetBinLabel(1, "HF-");
  summaryHist->GetYaxis()->SetBinLabel(2, "HF+");
  summaryHist->GetXaxis()->SetTitle("Wedge #");

  // Fill the report summary objects with default values, since these will only
  // be filled at the change of run.
  reportSummary_->Fill(1.0);

  for (unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX) {
    unsigned int iWedge = HLXHFMap[iHLX] + 1;
    unsigned int iEta = 2;
    if (iWedge >= 19) {
      iEta = 1;
      iWedge -= 18;
    }
    reportSummaryMap_->setBinContent(iWedge, iEta, 1.0);
  }
}

// ------------ method called to for each event  ------------
void HLXMonitor::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  while (HLXTCP.IsConnected() == false) {
    HLXTCP.SetIP(DistribIP1);
    if (HLXTCP.Connect() != 1) {
      std::cout << "Failed to connect to " << DistribIP1 << "." << std::endl;
      sleep(1);
      std::cout << "Trying " << DistribIP2 << std::endl;
      HLXTCP.SetIP(DistribIP2);
      if (HLXTCP.Connect() == 1)
        break;
      std::cout << "Failed to connect to " << DistribIP2 << "." << std::endl;
      std::cout << " Reconnect in " << reconnTime << " seconds." << std::endl;
      sleep(reconnTime);
    }
  }
  if (HLXTCP.IsConnected() == true) {
    std::cout << "Successfully connected." << std::endl;
  }

  if (HLXTCP.ReceiveLumiSection(lumiSection) == 1) {
    // If this is the first time through, set the runNumber ...
    if (runNumber_ != lumiSection.hdr.runNumber) {
      if (!currentRunEnded_ && runNumber_ != 0) {
        EndRun();
      }
      runNumber_ = lumiSection.hdr.runNumber;
      currentRunEnded_ = false;
      // std::cout << "Run number is: " << runNumber_ << std::endl;
      timeval startruntime;
      gettimeofday(&startruntime, nullptr);
      runStartTimeStamp_->Fill(getUTCtime(&startruntime));
    }

    // Fill the monitoring histograms
    FillHistograms(lumiSection);
    FillHistoHFCompare(lumiSection);
    FillEventInfo(lumiSection, iEvent);
    FillReportSummary();

    cout << "Run: " << lumiSection.hdr.runNumber << " Section: " << lumiSection.hdr.sectionNumber
         << " Orbit: " << lumiSection.hdr.startOrbit << endl;
    cout << "Et Lumi: " << lumiSection.lumiSummary.InstantETLumi << endl;
    cout << "Occ Lumi 1: " << lumiSection.lumiSummary.InstantOccLumi[0] << endl;
    cout << "Occ Lumi 2: " << lumiSection.lumiSummary.InstantOccLumi[1] << endl;
  } else {
    HLXTCP.Disconnect();
    EndRun();
  }
}

void HLXMonitor::EndRun() {
  FillReportSummary();

  // Do some things that should be done at the end of the run ...
  expectedNibbles_ = 0;
  for (unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX)
    totalNibbles_[iHLX] = 0;

  std::cout << "** Here in end run **" << std::endl;
  runNumber_ = 0;
  currentRunEnded_ = true;
  sectionInstantSumEt = 0;
  sectionInstantErrSumEt = 0;
  sectionInstantSumOcc1 = 0;
  sectionInstantErrSumOcc1 = 0;
  sectionInstantSumOcc2 = 0;
  sectionInstantErrSumOcc2 = 0;
  sectionInstantNorm = 0;
  lsBinOld = 0;
  lumiSectionCount = 0;
  previousSection = 0;
}

void HLXMonitor::FillHistograms(const LUMI_SECTION &section) {
  // Check for missing data
  if (previousSection != (section.hdr.sectionNumber - 1)) {
    double weight = (double)(section.hdr.sectionNumber - previousSection - 1);
    // std::cout << "Filling missing data! " << weight << std::endl;
    MissingDQMDataCheck->Fill(0.5, weight);
  }
  previousSection = section.hdr.sectionNumber;

  int lsBin = int(lumiSectionCount / num4NibblePerLS_);
  int lsBinBX = int(lumiSectionCount / num4NibblePerLS_);
  HistAvgLumiEtSum->Fill(lsBin, section.lumiSummary.InstantETLumi);
  HistAvgLumiOccSet1->Fill(lsBin, section.lumiSummary.InstantOccLumi[0]);
  HistAvgLumiOccSet2->Fill(lsBin, section.lumiSummary.InstantOccLumi[1]);

  int fillBin = lumiSectionCount + 1;
  if (fillBin > 128) {
    // If we are already more than 2 LS's in, move everything back by one bin
    // and fill the last bin with the new value.
    for (int iBin = 1; iBin < 128; ++iBin) {
      RecentInstantLumiEtSum->setBinContent(iBin, RecentInstantLumiEtSum->getBinContent(iBin + 1));
      RecentInstantLumiOccSet1->setBinContent(iBin, RecentInstantLumiOccSet1->getBinContent(iBin + 1));
      RecentInstantLumiOccSet2->setBinContent(iBin, RecentInstantLumiOccSet2->getBinContent(iBin + 1));
      RecentIntegratedLumiEtSum->setBinContent(iBin, RecentIntegratedLumiEtSum->getBinContent(iBin + 1));
      RecentIntegratedLumiOccSet1->setBinContent(iBin, RecentIntegratedLumiOccSet1->getBinContent(iBin + 1));
      RecentIntegratedLumiOccSet2->setBinContent(iBin, RecentIntegratedLumiOccSet2->getBinContent(iBin + 1));
    }
    fillBin = 128;
  }

  RecentInstantLumiEtSum->setBinContent(fillBin, section.lumiSummary.InstantETLumi);
  RecentInstantLumiEtSum->setBinError(fillBin, section.lumiSummary.InstantETLumiErr);
  RecentInstantLumiOccSet1->setBinContent(fillBin, section.lumiSummary.InstantOccLumi[0]);
  RecentInstantLumiOccSet1->setBinError(fillBin, section.lumiSummary.InstantOccLumiErr[0]);
  RecentInstantLumiOccSet2->setBinContent(fillBin, section.lumiSummary.InstantOccLumi[1]);
  RecentInstantLumiOccSet2->setBinError(fillBin, section.lumiSummary.InstantOccLumiErr[1]);

  double recentOldBinContent = RecentIntegratedLumiEtSum->getBinContent(fillBin - 1);
  if (fillBin == 1)
    recentOldBinContent = 0;
  double recentNewBinContent = recentOldBinContent + section.lumiSummary.InstantETLumi;
  RecentIntegratedLumiEtSum->setBinContent(fillBin, recentNewBinContent);
  recentOldBinContent = RecentIntegratedLumiOccSet1->getBinContent(fillBin - 1);
  if (fillBin == 1)
    recentOldBinContent = 0;
  recentNewBinContent = recentOldBinContent + section.lumiSummary.InstantOccLumi[0];
  RecentIntegratedLumiOccSet1->setBinContent(fillBin, recentNewBinContent);
  recentOldBinContent = RecentIntegratedLumiOccSet2->getBinContent(fillBin - 1);
  if (fillBin == 1)
    recentOldBinContent = 0;
  recentNewBinContent = recentOldBinContent + section.lumiSummary.InstantOccLumi[0];
  RecentIntegratedLumiOccSet2->setBinContent(fillBin, recentNewBinContent);

  double recentOldBinError = RecentIntegratedLumiEtSum->getBinError(fillBin - 1);
  if (fillBin == 1)
    recentOldBinError = 0;
  double recentNewBinError = sqrt(recentOldBinError * recentOldBinError +
                                  section.lumiSummary.InstantETLumiErr * section.lumiSummary.InstantETLumiErr);
  RecentIntegratedLumiEtSum->setBinError(fillBin, recentNewBinError);
  recentOldBinError = RecentIntegratedLumiOccSet1->getBinError(fillBin - 1);
  if (fillBin == 1)
    recentOldBinError = 0;
  recentNewBinError = sqrt(recentOldBinError * recentOldBinError +
                           section.lumiSummary.InstantOccLumiErr[0] * section.lumiSummary.InstantOccLumiErr[0]);
  RecentIntegratedLumiOccSet1->setBinError(fillBin, recentNewBinError);
  recentOldBinError = RecentIntegratedLumiOccSet2->getBinError(fillBin - 1);
  if (fillBin == 1)
    recentOldBinError = 0;
  recentNewBinError = sqrt(recentOldBinError * recentOldBinError +
                           section.lumiSummary.InstantOccLumiErr[1] * section.lumiSummary.InstantOccLumiErr[1]);
  RecentIntegratedLumiOccSet2->setBinError(fillBin, recentNewBinError);

  if (lsBinOld != lsBin) {
    HistInstantLumiEtSum->setBinContent(lsBin, sectionInstantSumEt);
    HistInstantLumiEtSum->setBinError(lsBin, sqrt(sectionInstantErrSumEt));
    HistInstantLumiOccSet1->setBinContent(lsBin, sectionInstantSumOcc1);
    HistInstantLumiOccSet1->setBinError(lsBin, sqrt(sectionInstantErrSumOcc1));
    HistInstantLumiOccSet2->setBinContent(lsBin, sectionInstantSumOcc2);
    HistInstantLumiOccSet2->setBinError(lsBin, sqrt(sectionInstantErrSumOcc2));

    double etDenom = fabs(sectionInstantSumEt);
    if (etDenom < 1e-10)
      etDenom = 1e-10;
    double occ1Denom = fabs(sectionInstantSumOcc1);
    if (occ1Denom < 1e-10)
      occ1Denom = 1e-10;
    double occ2Denom = fabs(sectionInstantSumOcc2);
    if (occ2Denom < 1e-10)
      occ2Denom = 1e-10;
    double etError = 100.0 * sqrt(sectionInstantErrSumEt) / etDenom;
    double occ1Error = 100.0 * sqrt(sectionInstantErrSumOcc1) / occ1Denom;
    double occ2Error = 100.0 * sqrt(sectionInstantErrSumOcc2) / occ2Denom;
    HistInstantLumiEtSumError->setBinContent(lsBinOld, etError);
    HistInstantLumiOccSet1Error->setBinContent(lsBinOld, occ1Error);
    HistInstantLumiOccSet2Error->setBinContent(lsBinOld, occ2Error);

    double histOldBinContent = HistIntegratedLumiEtSum->getBinContent(lsBinOld);
    if (lsBinOld == 0)
      histOldBinContent = 0;
    double histNewBinContent = histOldBinContent + sectionInstantSumEt;
    HistIntegratedLumiEtSum->setBinContent(lsBin, histNewBinContent);
    histOldBinContent = HistIntegratedLumiOccSet1->getBinContent(lsBinOld);
    if (lsBinOld == 0)
      histOldBinContent = 0;
    histNewBinContent = histOldBinContent + sectionInstantSumOcc1;
    HistIntegratedLumiOccSet1->setBinContent(lsBin, histNewBinContent);
    histOldBinContent = HistIntegratedLumiOccSet2->getBinContent(lsBinOld);
    if (lsBinOld == 0)
      histOldBinContent = 0;
    histNewBinContent = histOldBinContent + sectionInstantSumOcc2;
    HistIntegratedLumiOccSet2->setBinContent(lsBin, histNewBinContent);

    double histOldBinError = HistIntegratedLumiEtSum->getBinError(lsBinOld);
    if (lsBinOld == 0)
      histOldBinError = 0;
    double histNewBinError = sqrt(histOldBinError * histOldBinError + sectionInstantErrSumEt);
    HistIntegratedLumiEtSum->setBinError(lsBin, histNewBinError);
    histOldBinError = HistIntegratedLumiOccSet1->getBinError(lsBinOld);
    if (lsBinOld == 0)
      histOldBinError = 0;
    histNewBinError = sqrt(histOldBinError * histOldBinError + sectionInstantErrSumOcc1);
    HistIntegratedLumiOccSet1->setBinError(lsBin, histNewBinError);
    histOldBinError = HistIntegratedLumiOccSet2->getBinError(lsBinOld);
    if (lsBinOld == 0)
      histOldBinError = 0;
    histNewBinError = sqrt(histOldBinError * histOldBinError + sectionInstantErrSumOcc2);
    HistIntegratedLumiOccSet2->setBinError(lsBin, histNewBinError);

    sectionInstantSumEt = 0;
    sectionInstantErrSumEt = 0;
    sectionInstantSumOcc1 = 0;
    sectionInstantErrSumOcc1 = 0;
    sectionInstantSumOcc2 = 0;
    sectionInstantErrSumOcc2 = 0;
    sectionInstantNorm = 0;
    lsBinOld = lsBin;
  }

  sectionInstantSumEt += section.lumiSummary.InstantETLumi;
  sectionInstantErrSumEt += section.lumiSummary.InstantETLumiErr * section.lumiSummary.InstantETLumiErr;
  sectionInstantSumOcc1 += section.lumiSummary.InstantOccLumi[0];
  sectionInstantErrSumOcc1 += section.lumiSummary.InstantOccLumiErr[0] * section.lumiSummary.InstantOccLumiErr[0];
  sectionInstantSumOcc2 += section.lumiSummary.InstantOccLumi[1];
  sectionInstantErrSumOcc2 += section.lumiSummary.InstantOccLumiErr[1] * section.lumiSummary.InstantOccLumiErr[1];
  ++sectionInstantNorm;

  for (int iHLX = 0; iHLX < (int)NUM_HLX; ++iHLX) {
    unsigned int utotal1 = 0;
    unsigned int utotal2 = 0;
    unsigned int iWedge = HLXHFMap[iHLX];
    if (section.occupancy[iHLX].hdr.numNibbles != 0) {
      // Don't include the last one hundred BX in the average.
      for (unsigned int iBX = 0; iBX < NUM_BUNCHES; ++iBX) {
        // Normalize to number of towers
        unsigned int norm[2] = {0, 0};
        norm[0] += section.occupancy[iHLX].data[set1BelowIndex][iBX];
        norm[0] += section.occupancy[iHLX].data[set1BetweenIndex][iBX];
        norm[0] += section.occupancy[iHLX].data[set1AboveIndex][iBX];
        if (norm[0] == 0)
          norm[0] = 1;
        norm[1] += section.occupancy[iHLX].data[set2BelowIndex][iBX];
        norm[1] += section.occupancy[iHLX].data[set2BetweenIndex][iBX];
        norm[1] += section.occupancy[iHLX].data[set2AboveIndex][iBX];
        if (norm[1] == 0)
          norm[1] = 1;

        double normEt = section.etSum[iHLX].data[iBX] / (double)(norm[0] + norm[1]);
        double normOccSet1Below = (double)section.occupancy[iHLX].data[set1BelowIndex][iBX] / (double)norm[0];
        double normOccSet1Between = (double)section.occupancy[iHLX].data[set1BetweenIndex][iBX] / (double)norm[0];
        double normOccSet1Above = (double)section.occupancy[iHLX].data[set1AboveIndex][iBX] / (double)norm[0];
        double normOccSet2Below = (double)section.occupancy[iHLX].data[set2BelowIndex][iBX] / (double)norm[1];
        double normOccSet2Between = (double)section.occupancy[iHLX].data[set2BetweenIndex][iBX] / (double)norm[1];
        double normOccSet2Above = (double)section.occupancy[iHLX].data[set2AboveIndex][iBX] / (double)norm[1];

        // Averages & check sum
        if (iBX < NUM_BUNCHES - 100) {
          AvgEtSum->Fill(iWedge, normEt);

          AvgOccBelowSet1->Fill(iWedge, normOccSet1Below);
          AvgOccBetweenSet1->Fill(iWedge, normOccSet1Between);
          AvgOccAboveSet1->Fill(iWedge, normOccSet1Above);

          AvgOccBelowSet2->Fill(iWedge, normOccSet2Below);
          AvgOccBetweenSet2->Fill(iWedge, normOccSet2Between);
          AvgOccAboveSet2->Fill(iWedge, normOccSet2Above);

          if (iWedge < 18) {
            HistAvgEtSumHFP->Fill(lsBin, normEt);
            HistAvgOccBelowSet1HFP->Fill(lsBin, normOccSet1Below);
            HistAvgOccBetweenSet1HFP->Fill(lsBin, normOccSet1Between);
            HistAvgOccAboveSet1HFP->Fill(lsBin, normOccSet1Above);
            HistAvgOccBelowSet2HFP->Fill(lsBin, normOccSet2Below);
            HistAvgOccBetweenSet2HFP->Fill(lsBin, normOccSet2Between);
            HistAvgOccAboveSet2HFP->Fill(lsBin, normOccSet2Above);

            if (iBX >= (XMIN - 1) && iBX <= (XMAX - 1))
              BXvsTimeAvgEtSumHFP->Fill(lsBinBX, iBX, normEt / (num4NibblePerLS_ * 18.0 * 12.0));
          } else {
            HistAvgEtSumHFM->Fill(lsBin, normEt);
            HistAvgOccBelowSet1HFM->Fill(lsBin, normOccSet1Below);
            HistAvgOccBetweenSet1HFM->Fill(lsBin, normOccSet1Between);
            HistAvgOccAboveSet1HFM->Fill(lsBin, normOccSet1Above);
            HistAvgOccBelowSet2HFM->Fill(lsBin, normOccSet2Below);
            HistAvgOccBetweenSet2HFM->Fill(lsBin, normOccSet2Between);
            HistAvgOccAboveSet2HFM->Fill(lsBin, normOccSet2Above);

            if (iBX >= (XMIN - 1) && iBX <= (XMAX - 1))
              BXvsTimeAvgEtSumHFM->Fill(lsBinBX, iBX, normEt / (num4NibblePerLS_ * 18.0 * 12.0));
          }

          utotal1 += section.occupancy[iHLX].data[set1BelowIndex][iBX];
          utotal1 += section.occupancy[iHLX].data[set1BetweenIndex][iBX];
          utotal1 += section.occupancy[iHLX].data[set1AboveIndex][iBX];

          utotal2 += section.occupancy[iHLX].data[set2BelowIndex][iBX];
          utotal2 += section.occupancy[iHLX].data[set2BetweenIndex][iBX];
          utotal2 += section.occupancy[iHLX].data[set2AboveIndex][iBX];
        }

        if (Style == "BX") {
          // Get the correct bin ...
          TH1F *Set1BelowHist = Set1Below[iWedge]->getTH1F();
          int iBin = Set1BelowHist->FindBin((float)iBX);

          // Adjust the old bin content to make the new, unnormalize and
          // renormalize
          if (lumiSectionCount > 0) {
            double oldNormOccSet1Below = (Set1Below[iWedge]->getBinContent(iBin)) * (double)(lumiSectionCount);
            normOccSet1Below += oldNormOccSet1Below;
            normOccSet1Below /= (double)(lumiSectionCount + 1);
            double oldNormOccSet2Below = (Set2Below[iWedge]->getBinContent(iBin)) * (double)(lumiSectionCount);
            normOccSet2Below += oldNormOccSet2Below;
            normOccSet2Below /= (double)(lumiSectionCount + 1);

            double oldNormOccSet1Between = (Set1Between[iWedge]->getBinContent(iBin)) * (double)(lumiSectionCount);
            normOccSet1Between += oldNormOccSet1Between;
            normOccSet1Between /= (double)(lumiSectionCount + 1);
            double oldNormOccSet2Between = (Set2Between[iWedge]->getBinContent(iBin)) * (double)(lumiSectionCount);
            normOccSet2Between += oldNormOccSet2Between;
            normOccSet2Between /= (double)(lumiSectionCount + 1);

            double oldNormOccSet1Above = (Set1Above[iWedge]->getBinContent(iBin)) * (double)(lumiSectionCount);
            normOccSet1Above += oldNormOccSet1Above;
            normOccSet1Above /= (double)(lumiSectionCount + 1);
            double oldNormOccSet2Above = (Set2Above[iWedge]->getBinContent(iBin)) * (double)(lumiSectionCount);
            normOccSet2Above += oldNormOccSet2Above;
            normOccSet2Above /= (double)(lumiSectionCount + 1);

            double oldNormEt = ETSum[iWedge]->getBinContent(iBin) * (double)(lumiSectionCount);
            normEt += oldNormEt;
            normEt /= (double)(lumiSectionCount + 1);
          }
          Set1Below[iWedge]->setBinContent(iBin, normOccSet1Below);
          Set1Between[iWedge]->setBinContent(iBin, normOccSet1Between);
          Set1Above[iWedge]->setBinContent(iBin, normOccSet1Above);
          Set2Below[iWedge]->setBinContent(iBin, normOccSet2Below);
          Set2Between[iWedge]->setBinContent(iBin, normOccSet2Between);
          Set2Above[iWedge]->setBinContent(iBin, normOccSet2Above);
          ETSum[iWedge]->setBinContent(iBin, normEt);
        } else if (Style == "Dist") {
          Set1Below[iWedge]->Fill(normOccSet1Below);
          Set1Between[iWedge]->Fill(normOccSet1Between);
          Set1Above[iWedge]->Fill(normOccSet1Above);
          Set2Below[iWedge]->Fill(normOccSet2Below);
          Set2Between[iWedge]->Fill(normOccSet2Between);
          Set2Above[iWedge]->Fill(normOccSet2Above);
          ETSum[iWedge]->Fill(normEt);
        }
      }

      // Get the number of towers per wedge per BX (assuming non-zero numbers)
      double total1 = 0;
      double total2 = 0;
      if ((NUM_BUNCHES - 100) > 0) {
        total1 = (double)utotal1 / (double)(NUM_BUNCHES - 100);
        total2 = (double)utotal2 / (double)(NUM_BUNCHES - 100);
      }
      if (section.hdr.numOrbits > 0) {
        total1 = total1 / (double)section.hdr.numOrbits;
        total2 = total2 / (double)section.hdr.numOrbits;
      }

      SumAllOccSet1->Fill(iWedge, total1);
      SumAllOccSet2->Fill(iWedge, total2);
    }
  }

  double max[4] = {-1000.0, -1000.0, -1000.0, -1000.0};
  int bxmax[4] = {-1, -1, -1, -1};
  for (unsigned int iBX = 0; iBX < NUM_BUNCHES; ++iBX) {
    LumiAvgEtSum->Fill(iBX, section.lumiDetail.ETLumi[iBX]);
    LumiAvgOccSet1->Fill(iBX, section.lumiDetail.OccLumi[0][iBX]);
    LumiAvgOccSet2->Fill(iBX, section.lumiDetail.OccLumi[1][iBX]);

    if (section.lumiDetail.OccLumi[0][iBX] > max[0]) {
      max[3] = max[2];
      bxmax[3] = bxmax[2];
      max[2] = max[1];
      bxmax[2] = bxmax[1];
      max[1] = max[0];
      bxmax[1] = bxmax[0];
      max[0] = section.lumiDetail.OccLumi[0][iBX];
      bxmax[0] = iBX;
    } else if (section.lumiDetail.OccLumi[0][iBX] > max[1]) {
      max[3] = max[2];
      bxmax[3] = bxmax[2];
      max[2] = max[1];
      bxmax[2] = bxmax[1];
      max[1] = section.lumiDetail.OccLumi[0][iBX];
      bxmax[1] = iBX;
    } else if (section.lumiDetail.OccLumi[0][iBX] > max[2]) {
      max[3] = max[2];
      bxmax[3] = bxmax[2];
      max[2] = section.lumiDetail.OccLumi[0][iBX];
      bxmax[2] = iBX;
    } else if (section.lumiDetail.OccLumi[0][iBX] > max[3]) {
      max[3] = section.lumiDetail.OccLumi[0][iBX];
      bxmax[3] = iBX;
    }

    int iBin = iBX - (int)XMIN + 1;
    if (iBin <= int(XMAX - XMIN) && iBin >= 1) {
      LumiInstantEtSum->setBinContent(iBin, section.lumiDetail.ETLumi[iBX]);
      LumiInstantOccSet1->setBinContent(iBin, section.lumiDetail.OccLumi[0][iBX]);
      LumiInstantOccSet2->setBinContent(iBin, section.lumiDetail.OccLumi[1][iBX]);
      LumiInstantEtSum->setBinError(iBin, section.lumiDetail.ETLumiErr[iBX]);
      LumiInstantOccSet1->setBinError(iBin, section.lumiDetail.OccLumiErr[0][iBX]);
      LumiInstantOccSet2->setBinError(iBin, section.lumiDetail.OccLumiErr[1][iBX]);

      double oldBinContent = LumiIntegratedEtSum->getBinContent(iBin);
      if (lumiSectionCount == 0)
        oldBinContent = 0;
      double newBinContent = oldBinContent + section.lumiDetail.ETLumi[iBX];
      LumiIntegratedEtSum->setBinContent(iBin, newBinContent);
      oldBinContent = LumiIntegratedOccSet1->getBinContent(iBin);
      if (lumiSectionCount == 0)
        oldBinContent = 0;
      newBinContent = oldBinContent + section.lumiDetail.OccLumi[0][iBX];
      LumiIntegratedOccSet1->setBinContent(iBin, newBinContent);
      oldBinContent = LumiIntegratedOccSet2->getBinContent(iBin);
      if (lumiSectionCount == 0)
        oldBinContent = 0;
      newBinContent = oldBinContent + section.lumiDetail.OccLumi[1][iBX];
      LumiIntegratedOccSet2->setBinContent(iBin, newBinContent);

      double oldBinError = LumiIntegratedEtSum->getBinError(iBin);
      if (lumiSectionCount == 0)
        oldBinError = 0;
      double newBinError =
          sqrt(oldBinError * oldBinError + section.lumiDetail.ETLumiErr[iBX] * section.lumiDetail.ETLumiErr[iBX]);
      LumiIntegratedEtSum->setBinError(iBin, newBinError);
      oldBinError = LumiIntegratedOccSet1->getBinError(iBin);
      if (lumiSectionCount == 0)
        oldBinError = 0;
      newBinError = sqrt(oldBinError * oldBinError +
                         section.lumiDetail.OccLumiErr[0][iBX] * section.lumiDetail.OccLumiErr[0][iBX]);
      LumiIntegratedOccSet1->setBinError(iBin, newBinError);
      oldBinError = LumiIntegratedOccSet1->getBinError(iBin);
      if (lumiSectionCount == 0)
        oldBinError = 0;
      newBinError = sqrt(oldBinError * oldBinError +
                         section.lumiDetail.OccLumiErr[1][iBX] * section.lumiDetail.OccLumiErr[1][iBX]);
      LumiIntegratedOccSet2->setBinError(iBin, newBinError);
    }
  }

  // Now fill the maximum hists, but ordered by BX, so that
  // collision BX's or satellite BX's will always appear in the
  // same histogram.
  int flag = 1;
  for (int iM = 0; (iM < 4) && flag; ++iM) {
    flag = 0;
    for (int iN = 0; iN < 3; ++iN) {
      if (bxmax[iN + 1] < bxmax[iN]) {
        int tmp = bxmax[iN];
        bxmax[iN] = bxmax[iN + 1];
        bxmax[iN + 1] = tmp;

        double tmp2 = max[iN];
        max[iN] = max[iN + 1];
        max[iN + 1] = tmp2;
        flag = 1;
      }
    }
  }

  // 0.9e1 = Conversion constant for occ1 at 900GeV COM.
  MaxInstLumiBX1->Fill(max[0] * 0.9e1);
  MaxInstLumiBXNum1->Fill(bxmax[0]);
  MaxInstLumiBX2->Fill(max[1] * 0.9e1);
  MaxInstLumiBXNum2->Fill(bxmax[1]);
  MaxInstLumiBX3->Fill(max[2] * 0.9e1);
  MaxInstLumiBXNum3->Fill(bxmax[2]);
  MaxInstLumiBX4->Fill(max[3] * 0.9e1);
  MaxInstLumiBXNum4->Fill(bxmax[3]);

  TH1F *tmpHist = MaxInstLumiBX1->getTH1F();
  double minX = tmpHist->GetBinLowEdge(1);
  double maxX = tmpHist->GetBinLowEdge(tmpHist->GetNbinsX() + 1);

  int inum4NibblePerLS = (int)num4NibblePerLS_;
  if (lumiSectionCount % inum4NibblePerLS == 0) {
    double mean1 = MaxInstLumiBX1->getMean();
    double rms1 = MaxInstLumiBX1->getRMS();
    if (rms1 > 0 && mean1 - 5 * rms1 > minX && mean1 + 5 * rms1 < maxX)
      MaxInstLumiBX1->setAxisRange(mean1 - 5 * rms1, mean1 + 5 * rms1);

    double mean2 = MaxInstLumiBX2->getMean();
    double rms2 = MaxInstLumiBX2->getRMS();
    if (rms2 > 0 && mean2 - 5 * rms2 > minX && mean2 + 5 * rms2 < maxX)
      MaxInstLumiBX2->setAxisRange(mean2 - 5 * rms2, mean2 + 5 * rms2);

    double mean3 = MaxInstLumiBX3->getMean();
    double rms3 = MaxInstLumiBX3->getRMS();
    if (rms3 > 0 && mean3 - 5 * rms3 > minX && mean3 + 5 * rms3 < maxX)
      MaxInstLumiBX3->setAxisRange(mean3 - 5 * rms3, mean3 + 5 * rms3);

    double mean4 = MaxInstLumiBX4->getMean();
    double rms4 = MaxInstLumiBX4->getRMS();
    if (rms4 > 0 && mean4 - 5 * rms4 > minX && mean4 + 5 * rms4 < maxX)
      MaxInstLumiBX4->setAxisRange(mean4 - 5 * rms4, mean4 + 5 * rms4);
  }

  // Add one to the section count (usually short sections)
  ++lumiSectionCount;
}

void HLXMonitor::FillHistoHFCompare(const LUMI_SECTION &section) {
  for (unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX) {
    unsigned int iWedge = HLXHFMap[iHLX];

    if (section.occupancy[iHLX].hdr.numNibbles != 0) {
      float nActvTwrsSet1 = section.occupancy[iHLX].data[set1AboveIndex][TriggerBX] +
                            section.occupancy[iHLX].data[set1BetweenIndex][TriggerBX] +
                            section.occupancy[iHLX].data[set1BelowIndex][TriggerBX];

      float nActvTwrsSet2 = section.occupancy[iHLX].data[set2AboveIndex][TriggerBX] +
                            section.occupancy[iHLX].data[set2BetweenIndex][TriggerBX] +
                            section.occupancy[iHLX].data[set2BelowIndex][TriggerBX];

      float total = nActvTwrsSet1 + nActvTwrsSet2;

      if (total > 0) {
        float tempData = section.etSum[iHLX].data[TriggerBX] / total;
        // cout << "Filling HFCompare Et sum " << tempData << endl;
        HFCompareEtSum->Fill(iWedge, tempData);
      }

      if (nActvTwrsSet1 > 0) {
        float tempData = (float)section.occupancy[iHLX].data[set1BelowIndex][TriggerBX] / nActvTwrsSet1;
        HFCompareOccBelowSet1->Fill(iWedge, tempData);

        tempData = (float)section.occupancy[iHLX].data[set1BetweenIndex][TriggerBX] / nActvTwrsSet1;
        HFCompareOccBetweenSet1->Fill(iWedge, tempData);

        tempData = (float)section.occupancy[iHLX].data[set1AboveIndex][TriggerBX] / nActvTwrsSet1;
        HFCompareOccAboveSet1->Fill(iWedge, tempData);
      }

      if (nActvTwrsSet2 > 0) {
        float tempData = (float)section.occupancy[iHLX].data[set2BelowIndex][TriggerBX] / nActvTwrsSet2;
        HFCompareOccBelowSet2->Fill(iWedge, tempData);

        tempData = (float)section.occupancy[iHLX].data[set2BetweenIndex][TriggerBX] / nActvTwrsSet2;
        HFCompareOccBetweenSet2->Fill(iWedge, tempData);

        tempData = (float)section.occupancy[iHLX].data[set2AboveIndex][TriggerBX] / nActvTwrsSet2;
        HFCompareOccAboveSet2->Fill(iWedge, tempData);
      }
    }
  }
}

void HLXMonitor::FillEventInfo(const LUMI_SECTION &section, const edm::Event &e) {
  // New run .. set the run number and fill run summaries ...
  // std::cout << "Run number " << runNumber_ << " Section hdr run number "
  //	     << section.hdr.runNumber << std::endl;

  runId_->Fill(section.hdr.runNumber);
  lumisecId_->Fill((int)(section.hdr.sectionNumber / num4NibblePerLS_) + 1);

  // Update the total nibbles & the expected number
  expectedNibbles_ += section.hdr.numOrbits / 4096;
  for (unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX) {
    unsigned int iWedge = HLXHFMap[iHLX] + 1;
    totalNibbles_[iWedge - 1] += section.occupancy[iHLX].hdr.numNibbles;
  }

  eventId_->Fill(e.id().event());
  eventTimeStamp_->Fill(e.time().value() / (double)0xffffffff);

  pEvent_++;
  evtRateCount_++;
  processEvents_->Fill(pEvent_);

  lastUpdateTime_ = currentTime_;
  gettimeofday(&currentTime_, nullptr);
  processTimeStamp_->Fill(getUTCtime(&currentTime_));
  processLatency_->Fill(getUTCtime(&lastUpdateTime_, &currentTime_));

  float time = getUTCtime(&lastAvgTime_, &currentTime_);
  if (time >= (evtRateWindow_ * 60.0)) {
    processEventRate_->Fill((float)evtRateCount_ / time);
    evtRateCount_ = 0;
    lastAvgTime_ = currentTime_;
  }
}

void HLXMonitor::FillReportSummary() {
  // Run summary - Loop over the HLX's and fill the map,
  // also calculate the overall quality.
  float overall = 0.0;
  for (unsigned int iHLX = 0; iHLX < NUM_HLX; ++iHLX) {
    unsigned int iWedge = HLXHFMap[iHLX] + 1;
    unsigned int iEta = 2;
    float frac = 0.0;
    if (expectedNibbles_ > 0)
      frac = (float)totalNibbles_[iWedge - 1] / (float)expectedNibbles_;
    if (iWedge >= 19) {
      iEta = 1;
      iWedge -= 18;
    }
    reportSummaryMap_->setBinContent(iWedge, iEta, frac);
    overall += frac;
  }

  overall /= (float)NUM_HLX;
  if (overall > 1.0)
    overall = 0.0;
  // std::cout << "Filling report summary! Main. " << overall << std::endl;
  reportSummary_->Fill(overall);
}

double HLXMonitor::getUTCtime(timeval *a, timeval *b) {
  double deltaT = (*a).tv_sec * 1000.0 + (*a).tv_usec / 1000.0;
  if (b != nullptr)
    deltaT = (*b).tv_sec * 1000.0 + (*b).tv_usec / 1000.0 - deltaT;
  return deltaT / 1000.0;
}

// define this as a plug-in
DEFINE_FWK_MODULE(HLXMonitor);
