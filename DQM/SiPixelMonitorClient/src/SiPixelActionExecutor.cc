//#define printing false
//#define occupancyprinting false

#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <cmath>

#include <iostream>
using namespace std;
//=============================================================================================================
//
// -- Constructor
//
SiPixelActionExecutor::SiPixelActionExecutor(bool offlineXMLfile, bool Tier0Flag)
    : offlineXMLfile_(offlineXMLfile), Tier0Flag_(Tier0Flag) {
  edm::LogInfo("SiPixelActionExecutor") << " Creating SiPixelActionExecutor "
                                        << "\n";
  configParser_ = nullptr;
  configWriter_ = nullptr;
  ndet_ = 0;
  // collationDone = false;
}
//=============================================================================================================
//
// --  Destructor
//
SiPixelActionExecutor::~SiPixelActionExecutor() {
  edm::LogInfo("SiPixelActionExecutor") << " Deleting SiPixelActionExecutor "
                                        << "\n";
  if (configParser_)
    delete configParser_;
  if (configWriter_)
    delete configWriter_;
}
//=============================================================================================================
//
// -- Read Configuration File
//
void SiPixelActionExecutor::readConfiguration() {
  string localPath;
  if (offlineXMLfile_)
    localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else
    localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  if (configParser_ == nullptr) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
}
//=============================================================================================================
//
// -- Read Configuration File
//
bool SiPixelActionExecutor::readConfiguration(int &tkmap_freq,
                                              int &sum_barrel_freq,
                                              int &sum_endcap_freq,
                                              int &sum_grandbarrel_freq,
                                              int &sum_grandendcap_freq,
                                              int &message_limit_,
                                              int &source_type_,
                                              int &calib_type_) {
  // printing cout<<"Entering
  // SiPixelActionExecutor::readConfiguration..."<<endl;
  string localPath;
  if (offlineXMLfile_)
    localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else
    localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  if (configParser_ == nullptr) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }

  if (!configParser_->getFrequencyForTrackerMap(tkmap_freq)) {
    cout << "SiPixelActionExecutor::readConfiguration: Failed to read "
            "TrackerMap configuration parameters!! ";
    return false;
  }
  if (!configParser_->getFrequencyForBarrelSummary(sum_barrel_freq)) {
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Barrel Summary configuration parameters!! "
                                          << "\n";
    return false;
  }
  if (!configParser_->getFrequencyForEndcapSummary(sum_endcap_freq)) {
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Endcap Summary configuration parameters!! "
                                          << "\n";
    return false;
  }
  if (!configParser_->getFrequencyForGrandBarrelSummary(sum_grandbarrel_freq)) {
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Grand Barrel Summary configuration parameters!! "
                                          << "\n";
    return false;
  }
  if (!configParser_->getFrequencyForGrandEndcapSummary(sum_grandendcap_freq)) {
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Grand Endcap Summary configuration parameters!! "
                                          << "\n";
    return false;
  }
  if (!configParser_->getMessageLimitForQTests(message_limit_)) {
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read QTest Message Limit"
                                          << "\n";
    return false;
  }
  if (!configParser_->getSourceType(source_type_)) {
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Source Type"
                                          << "\n";
    return false;
  }
  if (!configParser_->getCalibType(calib_type_)) {
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Calib Type"
                                          << "\n";
    return false;
  }
  // printing cout<<"...leaving
  // SiPixelActionExecutor::readConfiguration..."<<endl;
  return true;
}
//=============================================================================================================
bool SiPixelActionExecutor::readConfiguration(int &tkmap_freq, int &summary_freq) {
  // printing cout<<"Entering
  // SiPixelActionExecutor::readConfiguration..."<<endl;
  string localPath;
  if (offlineXMLfile_)
    localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else
    localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  if (configParser_ == nullptr) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }

  if (!configParser_->getFrequencyForTrackerMap(tkmap_freq)) {
    cout << "SiPixelActionExecutor::readConfiguration: Failed to read "
            "TrackerMap configuration parameters!! ";
    return false;
  }
  if (!configParser_->getFrequencyForBarrelSummary(summary_freq)) {
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Summary configuration parameters!! "
                                          << "\n";
    return false;
  }
  // printing cout<<"...leaving
  // SiPixelActionExecutor::readConfiguration..."<<endl;
  return true;
}

//=============================================================================================================
void SiPixelActionExecutor::createSummary(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool isUpgrade) {
  // To be majorly overhauled and split into two, I guess.

  // cout<<"entering SiPixelActionExecutor::createSummary..."<<endl;
  string barrel_structure_name;
  vector<string> barrel_me_names;
  string localPath;
  if (offlineXMLfile_)
    localPath = string("DQM/SiPixelMonitorClient/test/sipixel_tier0_config.xml");
  else
    localPath = string("DQM/SiPixelMonitorClient/test/sipixel_monitorelement_config.xml");
  //  cout<<"*********************ATTENTION! LOCALPATH= "<<localPath<<endl;
  if (configParser_ == nullptr) {
    configParser_ = new SiPixelConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
  if (!configParser_->getMENamesForBarrelSummary(barrel_structure_name, barrel_me_names)) {
    cout << "SiPixelActionExecutor::createSummary: Failed to read Barrel "
            "Summary configuration parameters!! ";
    return;
  }
  configParser_->getSourceType(source_type_);

  iBooker.setCurrentFolder("Pixel/");
  iGetter.setCurrentFolder("Pixel/");
  fillSummary(iBooker, iGetter, barrel_structure_name, barrel_me_names, true,
              isUpgrade);  // Barrel
  iBooker.setCurrentFolder("Pixel/");
  iGetter.setCurrentFolder("Pixel/");
  string endcap_structure_name;
  vector<string> endcap_me_names;
  if (!configParser_->getMENamesForEndcapSummary(endcap_structure_name, endcap_me_names)) {
    edm::LogInfo("SiPixelActionExecutor") << "Failed to read Endcap Summary configuration parameters!! "
                                          << "\n";
    return;
  }

  //  cout << "--- Processing endcap" << endl;

  iBooker.setCurrentFolder("Pixel/");
  iGetter.setCurrentFolder("Pixel/");

  fillSummary(iBooker, iGetter, endcap_structure_name, endcap_me_names, false,
              isUpgrade);  // Endcap
  iBooker.setCurrentFolder("Pixel/");
  iGetter.setCurrentFolder("Pixel/");

  if (source_type_ == 0 || source_type_ == 5 || source_type_ == 20) {  // do this only if RawData source is present
    string federror_structure_name;
    vector<string> federror_me_names;
    if (!configParser_->getMENamesForFEDErrorSummary(federror_structure_name, federror_me_names)) {
      cout << "SiPixelActionExecutor::createSummary: Failed to read FED Error "
              "Summary configuration parameters!! ";
      return;
    }
    iBooker.setCurrentFolder("Pixel/");
    iGetter.setCurrentFolder("Pixel/");

    fillFEDErrorSummary(iBooker, iGetter, federror_structure_name, federror_me_names);
    iBooker.setCurrentFolder("Pixel/");
    iGetter.setCurrentFolder("Pixel/");
  }
  if (configWriter_)
    delete configWriter_;
  configWriter_ = nullptr;
  //  cout<<"leaving SiPixelActionExecutor::createSummary..."<<endl;
}

//=============================================================================================================
void SiPixelActionExecutor::bookDeviations(DQMStore::IBooker &iBooker, bool isUpgrade) {
  int nBPixModules;
  if (isUpgrade) {
    nBPixModules = 1184;
  } else {
    nBPixModules = 768;
  }

  iBooker.cd();
  iBooker.setCurrentFolder("Pixel/Barrel");
  DEV_adc_Barrel = iBooker.book1D(
      "DEV_adc_Barrel", "Deviation from reference;Module;<adc_ref>-<adc>", nBPixModules, 0., nBPixModules);
  DEV_ndigis_Barrel = iBooker.book1D(
      "DEV_ndigis_Barrel", "Deviation from reference;Module;<ndigis_ref>-<ndigis>", nBPixModules, 0., nBPixModules);
  DEV_charge_Barrel = iBooker.book1D(
      "DEV_charge_Barrel", "Deviation from reference;Module;<charge_ref>-<charge>", nBPixModules, 0., nBPixModules);
  DEV_nclusters_Barrel = iBooker.book1D("DEV_nclusters_Barrel",
                                        "Deviation from reference;Module;<nclusters_ref>-<nclusters>",
                                        nBPixModules,
                                        0.,
                                        nBPixModules);
  DEV_size_Barrel = iBooker.book1D(
      "DEV_size_Barrel", "Deviation from reference;Module;<size_ref>-<size>", nBPixModules, 0., nBPixModules);
  iBooker.cd();
  iBooker.setCurrentFolder("Pixel/Endcap");
  DEV_adc_Endcap = iBooker.book1D("DEV_adc_Endcap", "Deviation from reference;Module;<adc_ref>-<adc>", 672, 0., 672.);
  DEV_ndigis_Endcap =
      iBooker.book1D("DEV_ndigis_Endcap", "Deviation from reference;Module;<ndigis_ref>-<ndigis>", 672, 0., 672.);
  DEV_charge_Endcap =
      iBooker.book1D("DEV_charge_Endcap", "Deviation from reference;Module;<charge_ref>-<charge>", 672, 0., 672.);
  DEV_nclusters_Endcap = iBooker.book1D(
      "DEV_nclusters_Endcap", "Deviation from reference;Module;<nclusters_ref>-<nclusters>", 672, 0., 672.);
  DEV_size_Endcap =
      iBooker.book1D("DEV_size_Endcap", "Deviation from reference;Module;<size_ref>-<size>", 672, 0., 672.);
  iBooker.cd();
}

//=============================================================================================================

void SiPixelActionExecutor::GetBladeSubdirs(DQMStore::IBooker &iBooker,
                                            DQMStore::IGetter &iGetter,
                                            vector<string> &blade_subdirs) {
  blade_subdirs.clear();
  vector<string> panels = iGetter.getSubdirs();
  vector<string> modules;
  for (const auto &panel : panels) {
    iGetter.cd(panel);
    iBooker.cd(panel);
    modules = iGetter.getSubdirs();
    for (const auto &module : modules) {
      blade_subdirs.push_back(module);
    }
  }
}

//=============================================================================================================

void SiPixelActionExecutor::fillSummary(DQMStore::IBooker &iBooker,
                                        DQMStore::IGetter &iGetter,
                                        const string &dir_name,
                                        vector<string> &me_names,
                                        bool isbarrel,
                                        bool isUpgrade) {
  // cout<<"entering SiPixelActionExecutor::fillSummary..."<<endl;
  string currDir = iBooker.pwd();
  string prefix;
  if (source_type_ == 0)
    prefix = "SUMRAW";
  else if (source_type_ == 1)
    prefix = "SUMDIG";
  else if (source_type_ == 2)
    prefix = "SUMCLU";
  else if (source_type_ == 3)
    prefix = "SUMTRK";
  else if (source_type_ == 4)
    prefix = "SUMHIT";
  else if (source_type_ >= 7 && source_type_ < 20)
    prefix = "SUMCAL";
  else if (source_type_ == 20)
    prefix = "SUMOFF";
  if (currDir.find(dir_name) != string::npos) {
    vector<MonitorElement *> sum_mes;
    for (const auto &me_name : me_names) {
      if (source_type_ == 5 || source_type_ == 6) {
        if (me_name == "errorType" || me_name == "NErrors" || me_name == "fullType" || me_name == "chanNmbr" ||
            me_name == "TBMType" || me_name == "EvtNbr" || me_name == "evtSize" || me_name == "linkId" ||
            me_name == "ROCId" || me_name == "DCOLId" || me_name == "PXId" || me_name == "ROCNmbr" ||
            me_name == "TBMMessage" || me_name == "Type36Hitmap")
          prefix = "SUMRAW";
        else if (me_name == "ndigis" || me_name == "adc")
          prefix = "SUMDIG";
        else if (me_name == "nclusters" || me_name == "x" || me_name == "y" || me_name == "charge" ||
                 me_name == "size" || me_name == "sizeX" || me_name == "sizeY" || me_name == "minrow" ||
                 me_name == "maxrow" || me_name == "mincol" || me_name == "maxcol")
          prefix = "SUMCLU";
        if (currDir.find("Track") != string::npos)
          prefix = "SUMTRK";
        else if (me_name == "residualX" || me_name == "residualY")
          prefix = "SUMTRK";
        else if (me_name == "ClustX" || me_name == "ClustY" || me_name == "nRecHits" || me_name == "ErrorX" ||
                 me_name == "ErrorY")
          prefix = "SUMHIT";
        else if (me_name == "Gain1d" || me_name == "GainChi2NDF1d" || me_name == "GainChi2Prob1d" ||
                 me_name == "Pedestal1d" || me_name == "GainNPoints1d" || me_name == "GainHighPoint1d" ||
                 me_name == "GainLowPoint1d" || me_name == "GainEndPoint1d" || me_name == "GainFitResult2d" ||
                 me_name == "GainDynamicRange2d" || me_name == "GainSaturate2d" || me_name == "ScurveChi2ProbSummary" ||
                 me_name == "ScurveFitResultSummary" || me_name == "ScurveSigmasSummary" ||
                 me_name == "ScurveThresholdSummary" || me_name == "pixelAliveSummary" ||
                 me_name == "SiPixelErrorsCalibDigis")
          prefix = "SUMCAL";
      }
      MonitorElement *temp;
      string tag;
      if (me_name.find("residual") != string::npos) {  // track residuals
        tag = prefix + "_" + me_name + "_mean_" + currDir.substr(currDir.find(dir_name));
        temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
        sum_mes.push_back(temp);
        tag = prefix + "_" + me_name + "_RMS_" + currDir.substr(currDir.find(dir_name));
        temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
        sum_mes.push_back(temp);
      } else if (prefix == "SUMCAL") {  // calibrations
        if (me_name == "Gain1d" || me_name == "GainChi2NDF1d" || me_name == "GainChi2Prob1d" ||
            me_name == "GainNPoints1d" || me_name == "GainHighPoint1d" || me_name == "GainLowPoint1d" ||
            me_name == "GainEndPoint1d" || me_name == "GainDynamicRange2d" || me_name == "GainSaturate2d" ||
            me_name == "Pedestal1d" || me_name == "ScurveChi2ProbSummary" || me_name == "ScurveFitResultSummary" ||
            me_name == "ScurveSigmasSummary" || me_name == "ScurveThresholdSummary") {
          tag = prefix + "_" + me_name + "_mean_" + currDir.substr(currDir.find(dir_name));
          temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
          sum_mes.push_back(temp);
          tag = prefix + "_" + me_name + "_RMS_" + currDir.substr(currDir.find(dir_name));
          temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
          sum_mes.push_back(temp);
        } else if (me_name == "SiPixelErrorsCalibDigis") {
          tag = prefix + "_" + me_name + "_NCalibErrors_" + currDir.substr(currDir.find(dir_name));
          temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
          sum_mes.push_back(temp);
        } else if (me_name == "GainFitResult2d") {
          tag = prefix + "_" + me_name + "_NNegativeFits_" + currDir.substr(currDir.find(dir_name));
          temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
          sum_mes.push_back(temp);
        } else if (me_name == "pixelAliveSummary") {
          tag = prefix + "_" + me_name + "_FracOfPerfectPix_" + currDir.substr(currDir.find(dir_name));
          temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
          sum_mes.push_back(temp);
          tag = prefix + "_" + me_name + "_mean_" + currDir.substr(currDir.find(dir_name));
          temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
          sum_mes.push_back(temp);
        }
      } else {
        tag = prefix + "_" + me_name + "_" + currDir.substr(currDir.find(dir_name));
        temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
        sum_mes.push_back(temp);
        if (me_name == "ndigis") {
          tag = prefix + "_" + me_name + "FREQ_" + currDir.substr(currDir.find(dir_name));
          temp = getSummaryME(iBooker, iGetter, tag, isUpgrade);
          sum_mes.push_back(temp);
        }
        if (prefix == "SUMDIG" && me_name == "adc") {
          tag = "ALLMODS_" + me_name + "COMB_" + currDir.substr(currDir.find(dir_name));
          temp = nullptr;
          string fullpathname = iBooker.pwd() + "/" + tag;
          temp = iGetter.get(fullpathname);
          if (temp) {
            temp->Reset();
          } else {
            temp = iBooker.book1D(tag.c_str(), tag.c_str(), 128, 0., 256.);
          }
          sum_mes.push_back(temp);
        }
        if (prefix == "SUMCLU" && me_name == "charge") {
          tag = "ALLMODS_" + me_name + "COMB_" + currDir.substr(currDir.find(dir_name));
          temp = nullptr;
          string fullpathname = iBooker.pwd() + "/" + tag;
          temp = iGetter.get(fullpathname);
          if (temp) {
            temp->Reset();
          } else {
            temp = iBooker.book1D(tag.c_str(), tag.c_str(), 100, 0.,
                                  200.);  // To look to get the size automatically
          }
          sum_mes.push_back(temp);
        }
      }
    }
    if (sum_mes.empty()) {
      edm::LogInfo("SiPixelActionExecutor") << " Summary MEs can not be created"
                                            << "\n";
      return;
    }
    vector<string> subdirs = iGetter.getSubdirs();
    // Blade
    if (dir_name.find("Blade_") == 0)
      GetBladeSubdirs(iBooker, iGetter, subdirs);

    int ndet = 0;
    for (const auto &subdir : subdirs) {
      if (prefix != "SUMOFF" && subdir.find("Module_") == string::npos)
        continue;
      if (prefix == "SUMOFF" && subdir.find(isbarrel ? "Layer_" : "Disk_") == string::npos)
        continue;
      iBooker.cd(subdir);
      iGetter.cd(subdir);
      ndet++;

      vector<string> contents = iGetter.getMEs();

      for (auto sum_me : sum_mes) {
        for (const auto &content : contents) {
          string sname = (sum_me->getName());
          string tname = " ";
          tname = sname.substr(7, (sname.find('_', 7) - 6));
          if (sname.find("ALLMODS_adcCOMB_") != string::npos)
            tname = "adc_";
          if (sname.find("ALLMODS_chargeCOMB_") != string::npos)
            tname = "charge_";
          if (sname.find("_charge_") != string::npos && sname.find("Track_") == string::npos)
            tname = "charge_";
          if (sname.find("_nclusters_") != string::npos && sname.find("Track_") == string::npos)
            tname = "nclusters_";
          if (sname.find("_size_") != string::npos && sname.find("Track_") == string::npos)
            tname = "size_";
          if (sname.find("_charge_OffTrack_") != string::npos)
            tname = "charge_OffTrack_";
          if (sname.find("_nclusters_OffTrack_") != string::npos)
            tname = "nclusters_OffTrack_";
          if (sname.find("_size_OffTrack_") != string::npos)
            tname = "size_OffTrack_";
          if (sname.find("_sizeX_OffTrack_") != string::npos)
            tname = "sizeX_OffTrack_";
          if (sname.find("_sizeY_OffTrack_") != string::npos)
            tname = "sizeY_OffTrack_";
          if (sname.find("_charge_OnTrack_") != string::npos)
            tname = "charge_OnTrack_";
          if (sname.find("_nclusters_OnTrack_") != string::npos)
            tname = "nclusters_OnTrack_";
          if (sname.find("_size_OnTrack_") != string::npos)
            tname = "size_OnTrack_";
          if (sname.find("_sizeX_OnTrack_") != string::npos)
            tname = "sizeX_OnTrack_";
          if (sname.find("_sizeY_OnTrack_") != string::npos)
            tname = "sizeY_OnTrack_";
          if (tname.find("FREQ") != string::npos)
            tname = "ndigis_";
          if ((content).find(tname) == 0) {
            string fullpathname = iBooker.pwd() + "/" + content;
            MonitorElement *me = iGetter.get(fullpathname);

            if (me) {
              if (sname.find("_charge") != string::npos && sname.find("Track_") == string::npos &&
                  me->getName().find("Track_") != string::npos)
                continue;
              if (sname.find("_nclusters_") != string::npos && sname.find("Track_") == string::npos &&
                  me->getName().find("Track_") != string::npos)
                continue;
              if (sname.find("_size") != string::npos && sname.find("Track_") == string::npos &&
                  me->getName().find("Track_") != string::npos)
                continue;
              // fill summary histos:
              if (sname.find("_RMS_") != string::npos && sname.find("GainDynamicRange2d") == string::npos &&
                  sname.find("GainSaturate2d") == string::npos) {
                sum_me->Fill(ndet, me->getRMS());
              } else if (sname.find("GainDynamicRange2d") != string::npos ||
                         sname.find("GainSaturate2d") != string::npos) {
                float SumOfEntries = 0.;
                float SumOfSquaredEntries = 0.;
                int SumOfPixels = 0;
                for (int cols = 1; cols != me->getNbinsX() + 1; cols++)
                  for (int rows = 1; rows != me->getNbinsY() + 1; rows++) {
                    SumOfEntries += me->getBinContent(cols, rows);
                    SumOfSquaredEntries += (me->getBinContent(cols, rows)) * (me->getBinContent(cols, rows));
                    SumOfPixels++;
                  }

                float MeanInZ = SumOfEntries / float(SumOfPixels);
                float RMSInZ = sqrt(SumOfSquaredEntries / float(SumOfPixels));
                if (sname.find("_mean_") != string::npos)
                  sum_me->Fill(ndet, MeanInZ);
                if (sname.find("_RMS_") != string::npos)
                  sum_me->Fill(ndet, RMSInZ);
              } else if (sname.find("_FracOfPerfectPix_") != string::npos) {
                float nlast = me->getBinContent(me->getNbinsX());
                float nall = (me->getTH1F())->Integral(1, 11);
                sum_me->Fill(ndet, nlast / nall);
              } else if (sname.find("_NCalibErrors_") != string::npos || sname.find("FREQ_") != string::npos) {
                float nall = me->getEntries();
                sum_me->Fill(ndet, nall);
              } else if (sname.find("GainFitResult2d") != string::npos) {
                int NegFitPixels = 0;
                for (int cols = 1; cols != me->getNbinsX() + 1; cols++)
                  for (int rows = 1; rows != me->getNbinsY() + 1; rows++) {
                    if (me->getBinContent(cols, rows) < 0.)
                      NegFitPixels++;
                  }
                sum_me->Fill(ndet, float(NegFitPixels));
              } else if (sname.find("ALLMODS_adcCOMB_") != string::npos ||
                         (sname.find("ALLMODS_chargeCOMB_") != string::npos &&
                          me->getName().find("Track_") == string::npos)) {
                sum_me->getTH1F()->Add(me->getTH1F());
              } else if (sname.find("_NErrors_") != string::npos) {
                string path1 = fullpathname;
                path1 = path1.replace(path1.find("NErrors"), 7, "errorType");
                MonitorElement *me1 = iGetter.get(path1);
                bool notReset = true;
                if (me1) {
                  for (int jj = 1; jj < 16; jj++) {
                    if (me1->getBinContent(jj) > 0.) {
                      if (jj == 6) {  // errorType=30 (reset)
                        string path2 = path1;
                        path2 = path2.replace(path2.find("errorType"), 9, "TBMMessage");
                        MonitorElement *me2 = iGetter.get(path2);
                        if (me2)
                          if (me2->getBinContent(6) > 0. || me2->getBinContent(7) > 0.)
                            notReset = false;
                      }
                    }
                  }
                }
                if (notReset)
                  sum_me->Fill(ndet, me1->getEntries());
              } else if ((sname.find("_charge_") != string::npos && sname.find("Track_") == string::npos &&
                          me->getName().find("Track_") == string::npos) ||
                         (sname.find("_charge_") != string::npos && sname.find("_OnTrack_") != string::npos &&
                          me->getName().find("_OnTrack_") != string::npos) ||
                         (sname.find("_charge_") != string::npos && sname.find("_OffTrack_") != string::npos &&
                          me->getName().find("_OffTrack_") != string::npos) ||
                         (sname.find("_nclusters_") != string::npos && sname.find("Track_") == string::npos &&
                          me->getName().find("Track_") == string::npos) ||
                         (sname.find("_nclusters_") != string::npos && sname.find("_OnTrack_") != string::npos &&
                          me->getName().find("_OnTrack_") != string::npos) ||
                         (sname.find("_nclusters_") != string::npos && sname.find("_OffTrack_") != string::npos &&
                          me->getName().find("_OffTrack_") != string::npos) ||
                         (sname.find("_size") != string::npos && sname.find("Track_") == string::npos &&
                          me->getName().find("Track_") == string::npos) ||
                         (sname.find("_size") != string::npos && sname.find("_OnTrack_") != string::npos &&
                          me->getName().find("_OnTrack_") != string::npos) ||
                         (sname.find("_size") != string::npos && sname.find("_OffTrack_") != string::npos &&
                          me->getName().find("_OffTrack_") != string::npos)) {
                sum_me->Fill(ndet, me->getMean());
              } else if (sname.find("_charge_") == string::npos && sname.find("_nclusters_") == string::npos &&
                         sname.find("_size") == string::npos) {
                sum_me->Fill(ndet, me->getMean());
              }

              // set titles:
              if (prefix == "SUMOFF") {
                sum_me->setAxisTitle(isbarrel ? "Ladders" : "Blades", 1);
              } else if (sname.find("ALLMODS_adcCOMB_") != string::npos) {
                sum_me->setAxisTitle("Digi charge [ADC]", 1);
              } else if (sname.find("ALLMODS_chargeCOMB_") != string::npos) {
                sum_me->setAxisTitle("Cluster charge [kilo electrons]", 1);
              } else {
                sum_me->setAxisTitle("Modules", 1);
              }
              string title = " ";
              if (sname.find("_RMS_") != string::npos) {
                title = "RMS of " + sname.substr(7, (sname.find('_', 7) - 7)) + " per module";
              } else if (sname.find("_FracOfPerfectPix_") != string::npos) {
                title = "FracOfPerfectPix " + sname.substr(7, (sname.find('_', 7) - 7)) + " per module";
              } else if (sname.find("_NCalibErrors_") != string::npos) {
                title = "Number of CalibErrors " + sname.substr(7, (sname.find('_', 7) - 7)) + " per module";
              } else if (sname.find("_NNegativeFits_") != string::npos) {
                title = "Number of pixels with neg. fit result " + sname.substr(7, (sname.find('_', 7) - 7)) +
                        " per module";
              } else if (sname.find("FREQ_") != string::npos) {
                title = "NEvents with digis per module";
              } else if (sname.find("ALLMODS_adcCOMB_") != string::npos) {
                title = "NDigis";
              } else if (sname.find("ALLMODS_chargeCOMB_") != string::npos) {
                title = "NClusters";
              } else if (sname.find("_NErrors_") != string::npos) {
                if (prefix == "SUMOFF" && isbarrel)
                  title = "Total number of errors per Ladder";
                else if (prefix == "SUMOFF" && !isbarrel)
                  title = "Total number of errors per Blade";
                else
                  title = "Total number of errors per Module";
              } else {
                if (prefix == "SUMOFF")
                  title =
                      "Mean " + sname.substr(7, (sname.find('_', 7) - 7)) + (isbarrel ? " per Ladder" : " per Blade");
                else
                  title = "Mean " + sname.substr(7, (sname.find('_', 7) - 7)) + " per Module";
              }
              sum_me->setAxisTitle(title, 2);
            }
            break;
          }
        }
      }
      iBooker.goUp();
      iGetter.setCurrentFolder(iBooker.pwd());
      if (dir_name.find("Blade") == 0) {
        iBooker.goUp();  // Going up a second time if we are processing the Blade
        iGetter.setCurrentFolder(iBooker.pwd());
      }
    }  // end for it (subdirs)
  } else {
    vector<string> subdirs = iGetter.getSubdirs();
    // printing cout << "#\t" << iBooker.pwd() << endl;
    if (isbarrel) {
      for (const auto &subdir : subdirs) {
        //				 cout << "##\t" << iBooker.pwd() << "\t"
        //<<
        //(*it) << endl;
        if ((iBooker.pwd()).find("Endcap") != string::npos ||
            (iBooker.pwd()).find("AdditionalPixelErrors") != string::npos) {
          iBooker.goUp();
          iGetter.setCurrentFolder(iBooker.pwd());
        }
        iBooker.cd(subdir);
        iGetter.cd(subdir);
        if (subdir.find("Endcap") != string::npos || subdir.find("AdditionalPixelErrors") != string::npos)
          continue;
        fillSummary(iBooker, iGetter, dir_name, me_names, true,
                    isUpgrade);  // Barrel
        iBooker.goUp();
        iGetter.setCurrentFolder(iBooker.pwd());
      }
      string grandbarrel_structure_name;
      vector<string> grandbarrel_me_names;
      if (!configParser_->getMENamesForGrandBarrelSummary(grandbarrel_structure_name, grandbarrel_me_names)) {
        cout << "SiPixelActionExecutor::createSummary: Failed to read Grand "
                "Barrel Summary configuration parameters!! ";
        return;
      }
      fillGrandBarrelSummaryHistos(iBooker, iGetter, grandbarrel_me_names, isUpgrade);

    } else  // Endcap
    {
      for (const auto &subdir : subdirs) {
        if ((iBooker.pwd()).find("Barrel") != string::npos ||
            (iBooker.pwd()).find("AdditionalPixelErrors") != string::npos) {
          iBooker.goUp();
          iGetter.setCurrentFolder(iBooker.pwd());
        }
        iBooker.cd(subdir);
        iGetter.cd(subdir);
        if (subdir.find("Barrel") != string::npos || subdir.find("AdditionalPixelErrors") != string::npos)
          continue;
        fillSummary(iBooker, iGetter, dir_name, me_names, false,
                    isUpgrade);  // Endcap
        iBooker.goUp();
        iGetter.setCurrentFolder(iBooker.pwd());
      }
      string grandendcap_structure_name;
      vector<string> grandendcap_me_names;
      if (!configParser_->getMENamesForGrandEndcapSummary(grandendcap_structure_name, grandendcap_me_names)) {
        cout << "SiPixelActionExecutor::createSummary: Failed to read Grand "
                "Endcap Summary configuration parameters!! ";
        return;
      }
      fillGrandEndcapSummaryHistos(iBooker, iGetter, grandendcap_me_names, isUpgrade);
    }
  }
  //  cout<<"...leaving SiPixelActionExecutor::fillSummary!"<<endl;
}

//=============================================================================================================
void SiPixelActionExecutor::fillFEDErrorSummary(DQMStore::IBooker &iBooker,
                                                DQMStore::IGetter &iGetter,
                                                const string &dir_name,
                                                vector<string> &me_names) {
  // printing cout<<"entering
  // SiPixelActionExecutor::fillFEDErrorSummary..."<<endl;
  string currDir = iBooker.pwd();
  string prefix;
  if (source_type_ == 0)
    prefix = "SUMRAW";
  else if (source_type_ == 20)
    prefix = "SUMOFF";

  if (currDir.find(dir_name) != string::npos) {
    vector<MonitorElement *> sum_mes;
    for (const auto &me_name : me_names) {
      bool isBooked = false;
      vector<string> contents = iGetter.getMEs();
      for (const auto &content : contents)
        if (content.find(me_name) != string::npos)
          isBooked = true;
      if (source_type_ == 5 || source_type_ == 6) {
        if (me_name == "errorType" || me_name == "NErrors" || me_name == "fullType" || me_name == "chanNmbr" ||
            me_name == "TBMType" || me_name == "EvtNbr" || me_name == "evtSize" || me_name == "linkId" ||
            me_name == "ROCId" || me_name == "DCOLId" || me_name == "PXId" || me_name == "ROCNmbr" ||
            me_name == "TBMMessage" || me_name == "Type36Hitmap" || me_name == "FedChLErr" || me_name == "FedChNErr" ||
            me_name == "FedETypeNErr")
          prefix = "SUMRAW";
      }
      if (me_name == "errorType" || me_name == "NErrors" || me_name == "fullType" || me_name == "chanNmbr" ||
          me_name == "TBMType" || me_name == "EvtNbr" || me_name == "evtSize" || me_name == "linkId" ||
          me_name == "ROCId" || me_name == "DCOLId" || me_name == "PXId" || me_name == "ROCNmbr" ||
          me_name == "TBMMessage" || me_name == "Type36Hitmap") {
        string tag = prefix + "_" + me_name + "_FEDErrors";
        MonitorElement *temp = getFEDSummaryME(iBooker, iGetter, tag);
        sum_mes.push_back(temp);
      } else if (me_name == "FedChLErr" || me_name == "FedChNErr" || me_name == "FedETypeNErr") {
        string tag = prefix + "_" + me_name;
        MonitorElement *temp;
        if (me_name == "FedChLErr") {
          if (!isBooked)
            temp = iBooker.book2D("FedChLErr", "Type of last error", 40, -0.5, 39.5, 37, 0., 37.);
          else {
            string fullpathname = iBooker.pwd() + "/" + me_name;
            temp = iGetter.get(fullpathname);
            temp->Reset();
          }
        }  // If I don't reset this one, then I instead start adding error
        // codes..
        if (me_name == "FedChNErr") {
          if (!isBooked)
            temp = iBooker.book2D("FedChNErr", "Total number of errors", 40, -0.5, 39.5, 37, 0., 37.);
          else {
            string fullpathname = iBooker.pwd() + "/" + me_name;
            temp = iGetter.get(fullpathname);
            temp->Reset();
          }
        }  // If I don't reset this one, then I instead start adding error
        // codes..
        if (me_name == "FedETypeNErr") {
          if (!isBooked) {
            temp = iBooker.book2D("FedETypeNErr", "Number of each error type", 40, -0.5, 39.5, 21, 0., 21.);
            temp->setBinLabel(1, "ROC of 25", 2);
            temp->setBinLabel(2, "Gap word", 2);
            temp->setBinLabel(3, "Dummy word", 2);
            temp->setBinLabel(4, "FIFO full", 2);
            temp->setBinLabel(5, "Timeout", 2);
            temp->setBinLabel(6, "Stack full", 2);
            temp->setBinLabel(7, "Pre-cal issued", 2);
            temp->setBinLabel(8, "Trigger clear or sync", 2);
            temp->setBinLabel(9, "No token bit", 2);
            temp->setBinLabel(10, "Overflow", 2);
            temp->setBinLabel(11, "FSM error", 2);
            temp->setBinLabel(12, "Invalid #ROCs", 2);
            temp->setBinLabel(13, "Event number", 2);
            temp->setBinLabel(14, "Slink header", 2);
            temp->setBinLabel(15, "Slink trailer", 2);
            temp->setBinLabel(16, "Event size", 2);
            temp->setBinLabel(17, "Invalid channel#", 2);
            temp->setBinLabel(18, "ROC value", 2);
            temp->setBinLabel(19, "Dcol or pixel value", 2);
            temp->setBinLabel(20, "Readout order", 2);
            temp->setBinLabel(21, "CRC error", 2);
          } else {
            string fullpathname = iBooker.pwd() + "/" + me_name;
            temp = iGetter.get(fullpathname);
            temp->Reset();
          }  // If I don't reset this one, then I instead start adding error
             // codes..
        }
        sum_mes.push_back(temp);
      }
    }
    if (sum_mes.empty()) {
      edm::LogInfo("SiPixelActionExecutor") << " Summary MEs can not be created"
                                            << "\n";
      return;
    }
    vector<string> subdirs = iGetter.getSubdirs();
    int ndet = 0;
    for (const auto &subdir : subdirs) {
      if (subdir.find("FED_") == string::npos)
        continue;
      iBooker.cd(subdir);
      iGetter.cd(subdir);
      string fedid = subdir.substr(subdir.find("_") + 1);
      std::istringstream isst;
      isst.str(fedid);
      isst >> ndet;
      ndet++;
      vector<string> contents = iGetter.getMEs();

      for (auto sum_me : sum_mes) {
        for (const auto &content : contents) {
          if ((content.find("FedChNErr") != std::string::npos &&
               sum_me->getName().find("FedChNErr") != std::string::npos) ||
              (content.find("FedChLErr") != std::string::npos &&
               sum_me->getName().find("FedChLErr") != std::string::npos) ||
              (content.find("FedETypeNErr") != std::string::npos &&
               sum_me->getName().find("FedETypeNErr") != std::string::npos)) {
            string fullpathname = iBooker.pwd() + "/" + content;
            MonitorElement *me = iGetter.get(fullpathname);
            if (me) {
              for (int i = 0; i != 37; i++) {
                if (content.find("FedETypeNErr") != std::string::npos && i < 21)
                  sum_me->Fill(ndet - 1, i, me->getBinContent(i + 1));
                else
                  sum_me->Fill(ndet - 1, i, me->getBinContent(i + 1));
              }
            }
          }
          string sname = (sum_me->getName());
          string tname = " ";
          tname = sname.substr(7, (sname.find('_', 7) - 6));
          if ((content).find(tname) == 0) {
            string fullpathname = iBooker.pwd() + "/" + content;
            MonitorElement *me = iGetter.get(fullpathname);

            if (me) {
              if (me->getMean() > 0.) {
                if (sname.find("_NErrors_") != string::npos) {
                  string path1 = fullpathname;
                  path1 = path1.replace(path1.find("NErrors"), 7, "errorType");
                  MonitorElement *me1 = iGetter.get(path1);
                  bool notReset = true;
                  if (me1) {
                    for (int jj = 1; jj < 16; jj++) {
                      if (me1->getBinContent(jj) > 0.) {
                        if (jj == 6) {  // errorType=30 (reset)
                          string path2 = path1;
                          path2 = path2.replace(path2.find("errorType"), 9, "TBMMessage");
                          MonitorElement *me2 = iGetter.get(path2);
                          if (me2)
                            if (me2->getBinContent(6) > 0. || me2->getBinContent(7) > 0.)
                              notReset = false;
                        }
                      }
                    }
                  }
                  if (notReset)
                    sum_me->setBinContent(ndet, sum_me->getBinContent(ndet) + me1->getEntries());
                } else
                  sum_me->setBinContent(ndet, sum_me->getBinContent(ndet) + me->getEntries());
              }
              sum_me->setAxisTitle("FED #", 1);
              string title = " ";
              title = sname.substr(7, (sname.find('_', 7) - 7)) + " per FED";
              sum_me->setAxisTitle(title, 2);
            }
            break;
          }
        }
      }
      iBooker.goUp();
      iGetter.setCurrentFolder(iBooker.pwd());
    }
  } else {
    vector<string> subdirs = iGetter.getSubdirs();
    for (const auto &subdir : subdirs) {
      if (subdir.find("Endcap") != string::npos || subdir.find("Barrel") != string::npos)
        continue;
      iBooker.cd(subdir);
      iGetter.cd(subdir);
      fillFEDErrorSummary(iBooker, iGetter, dir_name, me_names);
      iBooker.goUp();
      iGetter.setCurrentFolder(iBooker.pwd());
    }
  }
  // printing cout<<"...leaving
  // SiPixelActionExecutor::fillFEDErrorSummary!"<<endl;
}

//=============================================================================================================
void SiPixelActionExecutor::fillGrandBarrelSummaryHistos(DQMStore::IBooker &iBooker,
                                                         DQMStore::IGetter &iGetter,
                                                         vector<string> &me_names,
                                                         bool isUpgrade) {
  //  cout<<"Entering
  //  SiPixelActionExecutor::fillGrandBarrelSummaryHistos...:"<<me_names.size()<<endl;
  vector<MonitorElement *> gsum_mes;
  string currDir = iBooker.pwd();
  string path_name = iBooker.pwd();
  string dir_name = path_name.substr(path_name.find_last_of('/') + 1);
  if ((dir_name.find("DQMData") == 0) || (dir_name.find("Pixel") == 0) ||
      (dir_name.find("AdditionalPixelErrors") == 0) || (dir_name.find("Endcap") == 0) ||
      (dir_name.find("HalfCylinder") == 0) || (dir_name.find("Disk") == 0) || (dir_name.find("Blade") == 0) ||
      (dir_name.find("Panel") == 0))
    return;
  vector<string> subdirs = iGetter.getSubdirs();
  int nDirs = subdirs.size();
  int iDir = 0;
  int nbin = 0;
  int nbin_i = 0;
  int nbin_subdir = 0;
  int cnt = 0;
  bool first_subdir = true;
  for (const auto &subdir : subdirs) {
    cnt++;
    iBooker.cd(subdir);
    iGetter.cd(subdir);
    vector<string> contents = iGetter.getMEs();

    iBooker.goUp();
    iGetter.setCurrentFolder(iBooker.pwd());

    string prefix;
    if (source_type_ == 0)
      prefix = "SUMRAW";
    else if (source_type_ == 1)
      prefix = "SUMDIG";
    else if (source_type_ == 2)
      prefix = "SUMCLU";
    else if (source_type_ == 3)
      prefix = "SUMTRK";
    else if (source_type_ == 4)
      prefix = "SUMHIT";
    else if (source_type_ >= 7 && source_type_ < 20)
      prefix = "SUMCAL";
    else if (source_type_ == 20)
      prefix = "SUMOFF";

    for (const auto &content : contents) {
      for (const auto &iv : me_names) {
        string var = "_" + iv + "_";
        if (content.find(var) != string::npos) {
          if ((var == "_charge_" || var == "_nclusters_" || var == "_size_" || var == "_sizeX_" || var == "_sizeY_") &&
              content.find("Track_") != string::npos)
            continue;
          string full_path = subdir + "/" + content;
          MonitorElement *me = iGetter.get(full_path);
          if (!me)
            continue;
          if (source_type_ == 5 || source_type_ == 6) {
            if (iv == "errorType" || iv == "NErrors" || iv == "fullType" || iv == "chanNmbr" || iv == "TBMType" ||
                iv == "EvtNbr" || iv == "evtSize" || iv == "linkId" || iv == "ROCId" || iv == "DCOLId" ||
                iv == "PXId" || iv == "ROCNmbr" || iv == "TBMMessage" || iv == "Type36Hitmap")
              prefix = "SUMRAW";
            else if (iv == "ndigis" || iv == "adc" || iv == "ndigisFREQ" || iv == "adcCOMB")
              prefix = "SUMDIG";
            else if (iv == "nclusters" || iv == "x" || iv == "y" || iv == "charge" || iv == "chargeCOMB" ||
                     iv == "size" || iv == "sizeX" || iv == "sizeY" || iv == "minrow" || iv == "maxrow" ||
                     iv == "mincol" || iv == "maxcol")
              prefix = "SUMCLU";
            if (currDir.find("Track") != string::npos)
              prefix = "SUMTRK";
            else if (iv == "residualX_mean" || iv == "residualY_mean" || iv == "residualX_RMS" || iv == "residualY_RMS")
              prefix = "SUMTRK";
            else if (iv == "ClustX" || iv == "ClustY" || iv == "nRecHits" || iv == "ErrorX" || iv == "ErrorY")
              prefix = "SUMHIT";
            else if (iv == "Gain1d_mean" || iv == "GainChi2NDF1d_mean" || iv == "GainChi2Prob1d_mean" ||
                     iv == "Pedestal1d_mean" || iv == "ScurveChi2ProbSummary_mean" ||
                     iv == "ScurveFitResultSummary_mean" || iv == "ScurveSigmasSummary_mean" ||
                     iv == "ScurveThresholdSummary_mean" || iv == "Gain1d_RMS" || iv == "GainChi2NDF1d_RMS" ||
                     iv == "GainChi2Prob1d_RMS" || iv == "Pedestal1d_RMS" || iv == "GainNPoints1d_mean" ||
                     iv == "GainNPoints1d_RMS" || iv == "GainHighPoint1d_mean" || iv == "GainHighPoint1d_RMS" ||
                     iv == "GainLowPoint1d_mean" || iv == "GainLowPoint1d_RMS" || iv == "GainEndPoint1d_mean" ||
                     iv == "GainEndPoint1d_RMS" || iv == "GainFitResult2d_mean" || iv == "GainFitResult2d_RMS" ||
                     iv == "GainDynamicRange2d_mean" || iv == "GainDynamicRange2d_RMS" || iv == "GainSaturate2d_mean" ||
                     iv == "GainSaturate2d_RMS" || iv == "ScurveChi2ProbSummary_RMS" ||
                     iv == "ScurveFitResultSummary_RMS" || iv == "ScurveSigmasSummary_RMS" ||
                     iv == "ScurveThresholdSummary_RMS" || iv == "pixelAliveSummary_mean" ||
                     iv == "pixelAliveSummary_FracOfPerfectPix" || iv == "SiPixelErrorsCalibDigis_NCalibErrors")
              prefix = "SUMCAL";
          }  // end source_type if

          if (first_subdir && !isUpgrade) {
            nbin = me->getTH1F()->GetNbinsX();
            string me_name = prefix + "_" + iv + "_" + dir_name;
            if (iv == "adcCOMB" || iv == "chargeCOMB")
              me_name = "ALLMODS_" + iv + "_" + dir_name;
            else if (prefix == "SUMOFF" && dir_name == "Barrel")
              nbin = 192;
            else if (iv == "adcCOMB")
              nbin = 256;
            else if (dir_name == "Barrel")
              nbin = 768;
            else if (prefix == "SUMOFF" && dir_name.find("Shell") != string::npos)
              nbin = 48;
            else if (dir_name.find("Shell") != string::npos)
              nbin = 192;
            else
              nbin = nbin * nDirs;

            getGrandSummaryME(iBooker, iGetter, nbin, me_name, gsum_mes);
          } else if (first_subdir && isUpgrade) {
            nbin = me->getTH1F()->GetNbinsX();
            string me_name = prefix + "_" + iv + "_" + dir_name;
            if (iv == "adcCOMB" || iv == "chargeCOMB")
              me_name = "ALLMODS_" + iv + "_" + dir_name;
            else if (prefix == "SUMOFF" && dir_name == "Barrel")
              nbin = 296;
            else if (iv == "adcCOMB")
              nbin = 256;
            else if (dir_name == "Barrel")
              nbin = 1184;
            else if (prefix == "SUMOFF" && dir_name.find("Shell") != string::npos)
              nbin = 74;
            else if (dir_name.find("Shell") != string::npos)
              nbin = 296;
            else
              nbin = nbin * nDirs;

            getGrandSummaryME(iBooker, iGetter, nbin, me_name, gsum_mes);
          }

          for (auto gsum_me : gsum_mes) {
            if (gsum_me->getName().find(var) != string::npos) {
              if (prefix == "SUMOFF")
                gsum_me->setAxisTitle("Ladders", 1);
              else if (gsum_me->getName().find("adcCOMB_") != string::npos)
                gsum_me->setAxisTitle("Digi charge [ADC]", 1);
              else if (gsum_me->getName().find("chargeCOMB_") != string::npos)
                gsum_me->setAxisTitle("Cluster charge [kilo electrons]", 1);
              else
                gsum_me->setAxisTitle("Modules", 1);

              // Setting title

              string title = "";
              if (gsum_me->getName().find("NErrors_") != string::npos && prefix == "SUMOFF")
                title = "Total number of errors per Ladder";
              else if (gsum_me->getName().find("NErrors_") != string::npos && prefix == "SUMRAW")
                title = "Total number of errors per Module";
              else if (prefix == "SUMOFF")
                title = "mean " + iv + " per Ladder";
              else if (gsum_me->getName().find("FREQ_") != string::npos && prefix != "SUMOFF")
                title = "NEvents with digis per Module";
              else if (gsum_me->getName().find("FREQ_") != string::npos && prefix == "SUMOFF")
                title = "NEvents with digis per Ladder/Blade";
              else if (gsum_me->getName().find("adcCOMB_") != string::npos)
                title = "NDigis";
              else if (gsum_me->getName().find("chargeCOMB_") != string::npos)
                title = "NClusters";
              else
                title = "mean " + iv + " per Module";
              gsum_me->setAxisTitle(title, 2);

              // Setting binning
              if (!isUpgrade) {
                if (gsum_me->getName().find("ALLMODS_adcCOMB_") != string::npos) {
                  nbin_subdir = 128;
                } else if (gsum_me->getName().find("ALLMODS_chargeCOMB_") != string::npos) {
                  nbin_subdir = 100;
                } else if (gsum_me->getName().find("Ladder") != string::npos) {
                  nbin_i = 0;
                  nbin_subdir = 4;
                } else if (gsum_me->getName().find("Layer") != string::npos) {
                  nbin_i = (cnt - 1) * 4;
                  nbin_subdir = 4;
                } else if (gsum_me->getName().find("Shell") != string::npos) {
                  if (prefix != "SUMOFF") {
                    if (iDir == 0) {
                      nbin_i = 0;
                      nbin_subdir = 40;
                    } else if (iDir == 1) {
                      nbin_i = 40;
                      nbin_subdir = 64;
                    } else if (iDir == 2) {
                      nbin_i = 104;
                      nbin_subdir = 88;
                    }
                  } else {
                    if (iDir == 0) {
                      nbin_i = 0;
                      nbin_subdir = 10;
                    } else if (iDir == 1) {
                      nbin_i = 10;
                      nbin_subdir = 16;
                    } else if (iDir == 2) {
                      nbin_i = 26;
                      nbin_subdir = 22;
                    }
                  }
                } else if (gsum_me->getName().find("Barrel") != string::npos) {
                  if (prefix != "SUMOFF") {
                    if (iDir == 0) {
                      nbin_i = 0;
                      nbin_subdir = 192;
                    } else if (iDir == 1) {
                      nbin_i = 192;
                      nbin_subdir = 192;
                    } else if (iDir == 2) {
                      nbin_i = 384;
                      nbin_subdir = 192;
                    } else if (iDir == 3) {
                      nbin_i = 576;
                      nbin_subdir = 192;
                    }
                  } else {
                    if (iDir == 0) {
                      nbin_i = 0;
                      nbin_subdir = 48;
                    } else if (iDir == 1) {
                      nbin_i = 48;
                      nbin_subdir = 48;
                    } else if (iDir == 2) {
                      nbin_i = 96;
                      nbin_subdir = 48;
                    } else if (iDir == 3) {
                      nbin_i = 144;
                      nbin_subdir = 48;
                    }
                  }
                }
              } else if (isUpgrade) {
                if (gsum_me->getName().find("ALLMODS_adcCOMB_") != string::npos) {
                  nbin_subdir = 128;
                } else if (gsum_me->getName().find("ALLMODS_chargeCOMB_") != string::npos) {
                  nbin_subdir = 100;
                } else if (gsum_me->getName().find("Ladder") != string::npos) {
                  nbin_i = 0;
                  nbin_subdir = 4;
                } else if (gsum_me->getName().find("Layer") != string::npos) {
                  nbin_i = (cnt - 1) * 4;
                  nbin_subdir = 4;
                } else if (gsum_me->getName().find("Shell") != string::npos) {
                  if (prefix != "SUMOFF") {
                    if (iDir == 0) {
                      nbin_i = 0;
                      nbin_subdir = 24;
                    }  // 40(2*20)-->24(2*12)
                    else if (iDir == 1) {
                      nbin_i = 24;
                      nbin_subdir = 56;
                    }  // 64(32*2)-->56(2*28)
                    else if (iDir == 2) {
                      nbin_i = 80;
                      nbin_subdir = 88;
                    }  // 88(44*2)-->same88(44*2)
                    else if (iDir == 3) {
                      nbin_i = 168;
                      nbin_subdir = 128;
                    }
                  } else {
                    if (iDir == 0) {
                      nbin_i = 0;
                      nbin_subdir = 6;
                    }  // 10-->6
                    else if (iDir == 1) {
                      nbin_i = 6;
                      nbin_subdir = 14;
                    }  // 16-->14
                    else if (iDir == 2) {
                      nbin_i = 20;
                      nbin_subdir = 22;
                    }  // 22-->same22
                    else if (iDir == 3) {
                      nbin_i = 42;
                      nbin_subdir = 32;
                    }
                  }
                } else if (gsum_me->getName().find("Barrel") != string::npos) {
                  if (prefix != "SUMOFF") {
                    if (iDir == 0) {
                      nbin_i = 0;
                      nbin_subdir = 296;
                    }  // 192=76 8/4-->296=1184/4
                    else if (iDir == 1) {
                      nbin_i = 296;
                      nbin_subdir = 296;
                    }  // 296*2,*3,*4=1184
                    else if (iDir == 2) {
                      nbin_i = 592;
                      nbin_subdir = 296;
                    } else if (iDir == 3) {
                      nbin_i = 888;
                      nbin_subdir = 296;
                    } else if (iDir == 4) {
                      nbin_i = 1184;
                      nbin_subdir = 296;
                    }
                  } else {
                    if (iDir == 0) {
                      nbin_i = 0;
                      nbin_subdir = 74;
                    }  // 48=192/4-->74=296/4
                    else if (iDir == 1) {
                      nbin_i = 74;
                      nbin_subdir = 74;
                    }  // 74*2,...*4=296
                    else if (iDir == 2) {
                      nbin_i = 148;
                      nbin_subdir = 74;
                    } else if (iDir == 3) {
                      nbin_i = 222;
                      nbin_subdir = 74;
                    } else if (iDir == 4) {
                      nbin_i = 296;
                      nbin_subdir = 74;
                    }
                  }
                }
              }

              if (gsum_me->getName().find("ndigisFREQ") == string::npos) {
                if ((gsum_me->getName().find("adcCOMB") != string::npos &&
                     me->getName().find("adcCOMB") != string::npos) ||
                    (gsum_me->getName().find("chargeCOMB") != string::npos &&
                     me->getName().find("chargeCOMB") != string::npos)) {
                  gsum_me->getTH1F()->Add(me->getTH1F());
                } else if ((gsum_me->getName().find("charge_") != string::npos &&
                            gsum_me->getName().find("Track_") == string::npos &&
                            me->getName().find("charge_") != string::npos &&
                            me->getName().find("Track_") == string::npos) ||
                           (gsum_me->getName().find("nclusters_") != string::npos &&
                            gsum_me->getName().find("Track_") == string::npos &&
                            me->getName().find("nclusters_") != string::npos &&
                            me->getName().find("Track_") == string::npos) ||
                           (gsum_me->getName().find("size_") != string::npos &&
                            gsum_me->getName().find("Track_") == string::npos &&
                            me->getName().find("size_") != string::npos &&
                            me->getName().find("Track_") == string::npos) ||
                           (gsum_me->getName().find("charge_OffTrack_") != string::npos &&
                            me->getName().find("charge_OffTrack_") != string::npos) ||
                           (gsum_me->getName().find("nclusters_OffTrack_") != string::npos &&
                            me->getName().find("nclusters_OffTrack_") != string::npos) ||
                           (gsum_me->getName().find("size_OffTrack_") != string::npos &&
                            me->getName().find("size_OffTrack_") != string::npos) ||
                           (gsum_me->getName().find("charge_OnTrack_") != string::npos &&
                            me->getName().find("charge_OnTrack_") != string::npos) ||
                           (gsum_me->getName().find("nclusters_OnTrack_") != string::npos &&
                            me->getName().find("nclusters_OnTrack_") != string::npos) ||
                           (gsum_me->getName().find("size_OnTrack_") != string::npos &&
                            me->getName().find("size_OnTrack_") != string::npos) ||
                           (gsum_me->getName().find("charge_") == string::npos &&
                            gsum_me->getName().find("nclusters_") == string::npos &&
                            gsum_me->getName().find("size_") == string::npos)) {
                  for (int k = 1; k < nbin_subdir + 1; k++)
                    if (me->getBinContent(k) > 0)
                      gsum_me->setBinContent(k + nbin_i, me->getBinContent(k));
                }
              } else if (me->getName().find("ndigisFREQ") != string::npos) {
                for (int k = 1; k < nbin_subdir + 1; k++)
                  if (me->getBinContent(k) > 0)
                    gsum_me->setBinContent(k + nbin_i, me->getBinContent(k));
              }
            }  // end var in igm (gsum_mes)
          }    // end igm loop
        }      // end var in im (contents)
      }        // end of iv loop
    }          // end of im loop
    iDir++;
    first_subdir = false;  // We are done processing the first directory, we
                           // don't add any new MEs in the future passes.
  }                        // end of it loop (subdirs)
  //  cout<<"...leaving
  //  SiPixelActionExecutor::fillGrandBarrelSummaryHistos!"<<endl;
}

//=============================================================================================================
void SiPixelActionExecutor::fillGrandEndcapSummaryHistos(DQMStore::IBooker &iBooker,
                                                         DQMStore::IGetter &iGetter,
                                                         vector<string> &me_names,
                                                         bool isUpgrade) {
  // printing cout<<"Entering
  // SiPixelActionExecutor::fillGrandEndcapSummaryHistos..."<<endl;
  vector<MonitorElement *> gsum_mes;
  string currDir = iBooker.pwd();
  string path_name = iBooker.pwd();
  string dir_name = path_name.substr(path_name.find_last_of('/') + 1);
  if ((dir_name.find("DQMData") == 0) || (dir_name.find("Pixel") == 0) ||
      (dir_name.find("AdditionalPixelErrors") == 0) || (dir_name.find("Barrel") == 0) ||
      (dir_name.find("Shell") == 0) || (dir_name.find("Layer") == 0) || (dir_name.find("Ladder") == 0))
    return;
  vector<string> subdirs = iGetter.getSubdirs();
  int iDir = 0;
  int nbin = 0;
  int nbin_i = 0;
  int nbin_subdir = 0;
  int cnt = 0;
  bool first_subdir = true;
  for (const auto &subdir : subdirs) {
    cnt++;
    iBooker.cd(subdir);
    iGetter.cd(subdir);
    vector<string> contents = iGetter.getMEs();
    iBooker.goUp();
    iGetter.setCurrentFolder(iBooker.pwd());

    string prefix;
    if (source_type_ == 0)
      prefix = "SUMRAW";
    else if (source_type_ == 1)
      prefix = "SUMDIG";
    else if (source_type_ == 2)
      prefix = "SUMCLU";
    else if (source_type_ == 3)
      prefix = "SUMTRK";
    else if (source_type_ == 4)
      prefix = "SUMHIT";
    else if (source_type_ >= 7 && source_type_ < 20)
      prefix = "SUMCAL";
    else if (source_type_ == 20)
      prefix = "SUMOFF";

    for (const auto &content : contents) {
      for (const auto &iv : me_names) {
        string var = "_" + iv + "_";
        if (content.find(var) != string::npos) {
          if ((var == "_charge_" || var == "_nclusters_" || var == "_size_" || var == "_sizeX_" || var == "_sizeY_") &&
              content.find("Track_") != string::npos)
            continue;
          string full_path = subdir + "/" + content;
          MonitorElement *me = iGetter.get(full_path);
          if (!me)
            continue;
          if (source_type_ == 5 || source_type_ == 6) {
            if (iv == "errorType" || iv == "NErrors" || iv == "fullType" || iv == "chanNmbr" || iv == "TBMType" ||
                iv == "EvtNbr" || iv == "evtSize" || iv == "linkId" || iv == "ROCId" || iv == "DCOLId" ||
                iv == "PXId" || iv == "ROCNmbr" || iv == "TBMMessage" || iv == "Type36Hitmap")
              prefix = "SUMRAW";
            else if (iv == "ndigis" || iv == "adc" || iv == "ndigisFREQ" || iv == "adcCOMB")
              prefix = "SUMDIG";
            else if (iv == "nclusters" || iv == "x" || iv == "y" || iv == "charge" || iv == "chargeCOMB" ||
                     iv == "size" || iv == "sizeX" || iv == "sizeY" || iv == "minrow" || iv == "maxrow" ||
                     iv == "mincol" || iv == "maxcol")
              prefix = "SUMCLU";
            if (currDir.find("Track") != string::npos)
              prefix = "SUMTRK";
            else if (iv == "residualX_mean" || iv == "residualY_mean" || iv == "residualX_RMS" || iv == "residualY_RMS")
              prefix = "SUMTRK";
            else if (iv == "ClustX" || iv == "ClustY" || iv == "nRecHits" || iv == "ErrorX" || iv == "ErrorY")
              prefix = "SUMHIT";
            else if (iv == "Gain1d_mean" || iv == "GainChi2NDF1d_mean" || iv == "GainChi2Prob1d_mean" ||
                     iv == "Pedestal1d_mean" || iv == "ScurveChi2ProbSummary_mean" ||
                     iv == "ScurveFitResultSummary_mean" || iv == "ScurveSigmasSummary_mean" ||
                     iv == "ScurveThresholdSummary_mean" || iv == "Gain1d_RMS" || iv == "GainChi2NDF1d_RMS" ||
                     iv == "GainChi2Prob1d_RMS" || iv == "Pedestal1d_RMS" || iv == "GainNPoints1d_mean" ||
                     iv == "GainNPoints1d_RMS" || iv == "GainHighPoint1d_mean" || iv == "GainHighPoint1d_RMS" ||
                     iv == "GainLowPoint1d_mean" || iv == "GainLowPoint1d_RMS" || iv == "GainEndPoint1d_mean" ||
                     iv == "GainEndPoint1d_RMS" || iv == "GainFitResult2d_mean" || iv == "GainFitResult2d_RMS" ||
                     iv == "GainDynamicRange2d_mean" || iv == "GainDynamicRange2d_RMS" || iv == "GainSaturate2d_mean" ||
                     iv == "GainSaturate2d_RMS" || iv == "ScurveChi2ProbSummary_RMS" ||
                     iv == "ScurveFitResultSummary_RMS" || iv == "ScurveSigmasSummary_RMS" ||
                     iv == "ScurveThresholdSummary_RMS" || iv == "pixelAliveSummary_mean" ||
                     iv == "pixelAliveSummary_FracOfPerfectPix" || iv == "SiPixelErrorsCalibDigis_NCalibErrors")
              prefix = "SUMCAL";
          }

          if (first_subdir && !isUpgrade) {
            nbin = me->getTH1F()->GetNbinsX();
            string me_name = prefix + "_" + iv + "_" + dir_name;
            if (iv == "adcCOMB" || iv == "chargeCOMB")
              me_name = "ALLMODS_" + iv + "_" + dir_name;
            else if (prefix == "SUMOFF" && dir_name == "Endcap")
              nbin = 96;
            else if (dir_name == "Endcap")
              nbin = 672;
            else if (prefix == "SUMOFF" && dir_name.find("HalfCylinder") != string::npos)
              nbin = 24;
            else if (dir_name.find("HalfCylinder") != string::npos)
              nbin = 168;
            else if (prefix == "SUMOFF" && dir_name.find("Disk") != string::npos)
              nbin = 12;
            else if (dir_name.find("Disk") != string::npos)
              nbin = 84;
            else if (dir_name.find("Blade") != string::npos)
              nbin = 7;
            getGrandSummaryME(iBooker, iGetter, nbin, me_name, gsum_mes);
          } else if (first_subdir && isUpgrade) {
            nbin = me->getTH1F()->GetNbinsX();
            string me_name = prefix + "_" + iv + "_" + dir_name;
            if (iv == "adcCOMB" || iv == "chargeCOMB")
              me_name = "ALLMODS_" + iv + "_" + dir_name;
            else if (prefix == "SUMOFF" && dir_name == "Endcap")
              nbin = 336;
            else if (dir_name == "Endcap")
              nbin = 672;
            else if (prefix == "SUMOFF" && dir_name.find("HalfCylinder") != string::npos)
              nbin = 84;
            else if (dir_name.find("HalfCylinder") != string::npos)
              nbin = 168;
            else if (prefix == "SUMOFF" && dir_name.find("Disk") != string::npos)
              nbin = 28;
            else if (dir_name.find("Disk") != string::npos)
              nbin = 56;
            else if (dir_name.find("Blade") != string::npos)
              nbin = 2;
            getGrandSummaryME(iBooker, iGetter, nbin, me_name, gsum_mes);
          }

          for (auto gsum_me : gsum_mes) {
            if (gsum_me->getName().find(var) != string::npos) {
              if (prefix == "SUMOFF")
                gsum_me->setAxisTitle("Blades", 1);
              else if (gsum_me->getName().find("adcCOMB_") != string::npos)
                gsum_me->setAxisTitle("Digi charge [ADC]", 1);
              else if (gsum_me->getName().find("chargeCOMB_") != string::npos)
                gsum_me->setAxisTitle("Cluster charge [kilo electrons]", 1);
              else
                gsum_me->setAxisTitle("Modules", 1);
              string title = "";
              if (gsum_me->getName().find("NErrors_") != string::npos && prefix == "SUMOFF")
                title = "Total number of errors per Blade";
              else if (gsum_me->getName().find("NErrors_") != string::npos && prefix == "SUMRAW")
                title = "Total number of errors per Module";
              else if (prefix == "SUMOFF")
                title = "mean " + iv + " per Blade";
              else if (gsum_me->getName().find("FREQ_") != string::npos)
                title = "NEvents with digis per Module";
              else if (gsum_me->getName().find("adcCOMB_") != string::npos)
                title = "NDigis";
              else if (gsum_me->getName().find("chargeCOMB_") != string::npos)
                title = "NClusters";
              else
                title = "mean " + iv + " per Module";
              gsum_me->setAxisTitle(title, 2);
              nbin_i = 0;
              if (!isUpgrade) {
                if (gsum_me->getName().find("ALLMODS_adcCOMB_") != string::npos) {
                  nbin_subdir = 128;
                } else if (gsum_me->getName().find("ALLMODS_chargeCOMB_") != string::npos) {
                  nbin_subdir = 100;
                } else if (gsum_me->getName().find("Panel_") != string::npos) {
                  nbin_subdir = 7;
                } else if (gsum_me->getName().find("Blade") != string::npos) {
                  if (content.find("_1") != string::npos)
                    nbin_subdir = 4;
                  if (content.find("_2") != string::npos) {
                    nbin_i = 4;
                    nbin_subdir = 3;
                  }
                } else if (gsum_me->getName().find("Disk") != string::npos) {
                  nbin_i = ((cnt - 1) % 12) * 7;
                  nbin_subdir = 7;
                } else if (gsum_me->getName().find("HalfCylinder") != string::npos) {
                  if (prefix != "SUMOFF") {
                    nbin_subdir = 84;
                    if (content.find("_2") != string::npos)
                      nbin_i = 84;
                  } else {
                    nbin_subdir = 12;
                    if (content.find("_2") != string::npos)
                      nbin_i = 12;
                  }
                } else if (gsum_me->getName().find("Endcap") != string::npos) {
                  if (prefix != "SUMOFF") {
                    nbin_subdir = 168;
                    if (content.find("_mO") != string::npos)
                      nbin_i = 168;
                    if (content.find("_pI") != string::npos)
                      nbin_i = 336;
                    if (content.find("_pO") != string::npos)
                      nbin_i = 504;
                  } else {
                    nbin_subdir = 24;
                    if (content.find("_mO") != string::npos)
                      nbin_i = 24;
                    if (content.find("_pI") != string::npos)
                      nbin_i = 48;
                    if (content.find("_pO") != string::npos)
                      nbin_i = 72;
                  }
                }
              } else if (isUpgrade) {
                if (gsum_me->getName().find("ALLMODS_adcCOMB_") != string::npos) {
                  nbin_subdir = 128;
                } else if (gsum_me->getName().find("ALLMODS_chargeCOMB_") != string::npos) {
                  nbin_subdir = 100;
                } else if (gsum_me->getName().find("Panel_") != string::npos) {
                  nbin_subdir = 2;
                } else if (gsum_me->getName().find("Blade") != string::npos) {
                  if (content.find("_1") != string::npos)
                    nbin_subdir = 1;
                  if (content.find("_2") != string::npos) {
                    nbin_i = 1;
                    nbin_subdir = 1;
                  }
                } else if (gsum_me->getName().find("Disk") != string::npos) {
                  nbin_i = ((cnt - 1) % 28) * 2;
                  nbin_subdir = 2;
                } else if (gsum_me->getName().find("HalfCylinder") != string::npos) {
                  if (prefix != "SUMOFF") {
                    nbin_subdir = 56;
                    if (content.find("_2") != string::npos)
                      nbin_i = 56;
                    if (content.find("_3") != string::npos)
                      nbin_i = 112;
                  } else {
                    nbin_subdir = 28;
                    if (content.find("_2") != string::npos)
                      nbin_i = 28;
                    if (content.find("_3") != string::npos)
                      nbin_i = 56;
                  }
                } else if (gsum_me->getName().find("Endcap") != string::npos) {
                  if (prefix != "SUMOFF") {
                    nbin_subdir = 168;
                    if (content.find("_mO") != string::npos)
                      nbin_i = 168;
                    if (content.find("_pI") != string::npos)
                      nbin_i = 336;
                    if (content.find("_pO") != string::npos)
                      nbin_i = 504;
                  } else {
                    nbin_subdir = 84;
                    if (content.find("_mO") != string::npos)
                      nbin_i = 84;
                    if (content.find("_pI") != string::npos)
                      nbin_i = 168;
                    if (content.find("_pO") != string::npos)
                      nbin_i = 252;
                  }
                }
              }

              if (gsum_me->getName().find("ndigisFREQ") == string::npos) {
                if ((gsum_me->getName().find("adcCOMB") != string::npos &&
                     me->getName().find("adcCOMB") != string::npos) ||
                    (gsum_me->getName().find("chargeCOMB") != string::npos &&
                     me->getName().find("chargeCOMB") != string::npos)) {
                  gsum_me->getTH1F()->Add(me->getTH1F());
                } else if ((gsum_me->getName().find("charge_") != string::npos &&
                            gsum_me->getName().find("Track_") == string::npos &&
                            me->getName().find("charge_") != string::npos &&
                            me->getName().find("Track_") == string::npos) ||
                           (gsum_me->getName().find("nclusters_") != string::npos &&
                            gsum_me->getName().find("Track_") == string::npos &&
                            me->getName().find("nclusters_") != string::npos &&
                            me->getName().find("Track_") == string::npos) ||
                           (gsum_me->getName().find("size_") != string::npos &&
                            gsum_me->getName().find("Track_") == string::npos &&
                            me->getName().find("size_") != string::npos &&
                            me->getName().find("Track_") == string::npos) ||
                           (gsum_me->getName().find("charge_OffTrack_") != string::npos &&
                            me->getName().find("charge_OffTrack_") != string::npos) ||
                           (gsum_me->getName().find("nclusters_OffTrack_") != string::npos &&
                            me->getName().find("nclusters_OffTrack_") != string::npos) ||
                           (gsum_me->getName().find("size_OffTrack_") != string::npos &&
                            me->getName().find("size_OffTrack_") != string::npos) ||
                           (gsum_me->getName().find("charge_OnTrack_") != string::npos &&
                            me->getName().find("charge_OnTrack_") != string::npos) ||
                           (gsum_me->getName().find("nclusters_OnTrack_") != string::npos &&
                            me->getName().find("nclusters_OnTrack_") != string::npos) ||
                           (gsum_me->getName().find("size_OnTrack_") != string::npos &&
                            me->getName().find("size_OnTrack_") != string::npos) ||
                           (gsum_me->getName().find("charge_") == string::npos &&
                            gsum_me->getName().find("nclusters_") == string::npos &&
                            gsum_me->getName().find("size_") == string::npos)) {
                  for (int k = 1; k < nbin_subdir + 1; k++)
                    if (me->getBinContent(k) > 0)
                      gsum_me->setBinContent(k + nbin_i, me->getBinContent(k));
                }
              } else if (me->getName().find("ndigisFREQ") != string::npos) {
                for (int k = 1; k < nbin_subdir + 1; k++)
                  if (me->getBinContent(k) > 0)
                    gsum_me->setBinContent(k + nbin_i, me->getBinContent(k));
              }
              //	       }// for
            }
          }
        }
      }
    }

    iDir++;
    first_subdir = false;  // We are done processing the first directory, we
                           // don't add any new MEs in the future passes.
  }                        // end for it (subdirs)
}
//=============================================================================================================
//
// -- Get Summary ME
//
void SiPixelActionExecutor::getGrandSummaryME(
    DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, int nbin, string &me_name, vector<MonitorElement *> &mes) {
  // printing cout<<"Entering SiPixelActionExecutor::getGrandSummaryME for:
  // "<<me_name<<endl;
  if ((iBooker.pwd()).find("Pixel") == string::npos)
    return;  // If one doesn't find pixel
  vector<string> contents = iGetter.getMEs();

  for (const auto &content : contents) {
    // printing cout<<"in grand summary me: "<<me_name<<","<<(*it)<<endl;
    if (content.find(me_name) == 0) {
      string fullpathname = iBooker.pwd() + "/" + me_name;
      MonitorElement *me = iGetter.get(fullpathname);

      if (me) {
        me->Reset();
        mes.push_back(me);
        return;
      }
    }
  }

  MonitorElement *temp_me(nullptr);
  if (me_name.find("ALLMODS_adcCOMB_") != string::npos)
    temp_me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 128, 0, 256);
  else if (me_name.find("ALLMODS_chargeCOMB_") != string::npos)
    temp_me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 100, 0, 200);
  else
    temp_me = iBooker.book1D(me_name.c_str(), me_name.c_str(), nbin, 1., nbin + 1.);
  if (temp_me)
    mes.push_back(temp_me);

  //  if(temp_me) cout<<"finally found grand ME: "<<me_name<<endl;
}

//=============================================================================================================
//
// -- Get Summary ME
//
SiPixelActionExecutor::MonitorElement *SiPixelActionExecutor::getSummaryME(DQMStore::IBooker &iBooker,
                                                                           DQMStore::IGetter &iGetter,
                                                                           const string &me_name,
                                                                           bool isUpgrade) {
  // printing cout<<"Entering SiPixelActionExecutor::getSummaryME for:
  // "<<me_name<<endl;
  MonitorElement *me = nullptr;
  if ((iBooker.pwd()).find("Pixel") == string::npos)
    return me;
  vector<string> contents = iGetter.getMEs();

  for (const auto &content : contents) {
    if (content.find(me_name) == 0) {
      string fullpathname = iBooker.pwd() + "/" + content;
      me = iGetter.get(fullpathname);
      if (me) {
        me->Reset();
        return me;
      }
    }
  }
  contents.clear();
  if (!isUpgrade) {
    if (me_name.find("SUMOFF") == string::npos) {
      if (me_name.find("Blade_") != string::npos)
        me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 7, 1., 8.);
      else
        me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 4, 1., 5.);
    } else if (me_name.find("Layer_1") != string::npos) {
      me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 10, 1., 11.);
    } else if (me_name.find("Layer_2") != string::npos) {
      me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 16, 1., 17.);
    } else if (me_name.find("Layer_3") != string::npos) {
      me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 22, 1., 23.);
    } else if (me_name.find("Disk_") != string::npos) {
      me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 12, 1., 13.);
    }
  }  // endifNOTUpgrade
  else if (isUpgrade) {
    if (me_name.find("SUMOFF") == string::npos) {
      if (me_name.find("Blade_") != string::npos)
        me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 2, 1., 3.);
      else
        me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 1, 1., 2.);
    } else if (me_name.find("Layer_1") != string::npos) {
      me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 6, 1., 7.);
    } else if (me_name.find("Layer_2") != string::npos) {
      me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 14, 1., 15.);
    } else if (me_name.find("Layer_3") != string::npos) {
      me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 22, 1., 23.);
    } else if (me_name.find("Layer_4") != string::npos) {
      me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 32, 1., 33.);
    } else if (me_name.find("Disk_") != string::npos) {
      me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 28, 1., 29.);
    }
  }  // endifUpgrade

  return me;
}

//=============================================================================================================
SiPixelActionExecutor::MonitorElement *SiPixelActionExecutor::getFEDSummaryME(DQMStore::IBooker &iBooker,
                                                                              DQMStore::IGetter &iGetter,
                                                                              const string &me_name) {
  // printing cout<<"Entering SiPixelActionExecutor::getFEDSummaryME..."<<endl;
  MonitorElement *me = nullptr;
  if ((iBooker.pwd()).find("Pixel") == string::npos)
    return me;
  vector<string> contents = iGetter.getMEs();

  for (const auto &content : contents) {
    if (content.find(me_name) == 0) {
      string fullpathname = iBooker.pwd() + "/" + content;

      me = iGetter.get(fullpathname);

      if (me) {
        me->Reset();
        return me;
      }
    }
  }
  contents.clear();
  me = iBooker.book1D(me_name.c_str(), me_name.c_str(), 40, -0.5, 39.5);

  return me;
}

//=============================================================================================================
void SiPixelActionExecutor::bookOccupancyPlots(DQMStore::IBooker &iBooker,
                                               DQMStore::IGetter &iGetter,
                                               bool hiRes,
                                               bool isbarrel)  // Polymorphism
{
  if (Tier0Flag_)
    return;
  vector<string> subdirs = iGetter.getSubdirs();
  for (const auto &subdir : subdirs) {
    if (isbarrel && subdir.find("Barrel") == string::npos)
      continue;
    if (!isbarrel && subdir.find("Endcap") == string::npos)
      continue;

    if (subdir.find("Module_") != string::npos)
      continue;
    if (subdir.find("Panel_") != string::npos)
      continue;
    if (subdir.find("Ladder_") != string::npos)
      continue;
    if (subdir.find("Blade_") != string::npos)
      continue;
    if (subdir.find("Layer_") != string::npos)
      continue;
    if (subdir.find("Disk_") != string::npos)
      continue;
    iBooker.cd(subdir);
    iGetter.cd(subdir);
    bookOccupancyPlots(iBooker, iGetter, hiRes, isbarrel);
    if (!hiRes) {
      // occupancyprinting cout<<"booking low res barrel occ plot now!"<<endl;
      OccupancyMap = iBooker.book2D((isbarrel ? "barrelOccupancyMap" : "endcapOccupancyMap"),
                                    "Barrel Digi Occupancy Map (4 pix per bin)",
                                    isbarrel ? 208 : 130,
                                    0.,
                                    isbarrel ? 416. : 260.,
                                    80,
                                    0.,
                                    160.);
    } else {
      // occupancyprinting cout<<"booking high res barrel occ plot now!"<<endl;
      OccupancyMap = iBooker.book2D((isbarrel ? "barrelOccupancyMap" : "endcapOccupancyMap"),
                                    "Barrel Digi Occupancy Map (1 pix per bin)",
                                    isbarrel ? 416 : 260,
                                    0.,
                                    isbarrel ? 416. : 260.,
                                    160,
                                    0.,
                                    160.);
    }
    OccupancyMap->setAxisTitle("Columns", 1);
    OccupancyMap->setAxisTitle("Rows", 2);

    iBooker.goUp();
    iGetter.setCurrentFolder(iBooker.pwd());
  }
}
//=============================================================================================================
void SiPixelActionExecutor::bookOccupancyPlots(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool hiRes) {
  if (Tier0Flag_)
    return;
  // Barrel
  iGetter.cd();
  iBooker.cd();
  iGetter.setCurrentFolder("Pixel");
  iBooker.setCurrentFolder("Pixel");
  this->bookOccupancyPlots(iBooker, iGetter, hiRes, true);

  // Endcap
  iGetter.cd();
  iBooker.cd();
  iGetter.setCurrentFolder("Pixel");
  iBooker.setCurrentFolder("Pixel");
  this->bookOccupancyPlots(iBooker, iGetter, hiRes, false);
}

void SiPixelActionExecutor::createOccupancy(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) {
  // std::cout<<"entering SiPixelActionExecutor::createOccupancy..."<<std::endl;
  if (Tier0Flag_)
    return;
  iBooker.cd();
  iGetter.cd();
  fillOccupancy(iBooker, iGetter, true);
  iBooker.cd();
  iGetter.cd();
  fillOccupancy(iBooker, iGetter, false);
  iBooker.cd();
  iGetter.cd();

  // std::cout<<"leaving SiPixelActionExecutor::createOccupancy..."<<std::endl;
}

//=============================================================================================================

void SiPixelActionExecutor::fillOccupancy(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool isbarrel) {
  // occupancyprinting cout<<"entering
  // SiPixelActionExecutor::fillOccupancy..."<<std::endl;
  if (Tier0Flag_)
    return;
  string currDir = iBooker.pwd();
  string dname = currDir.substr(currDir.find_last_of('/') + 1);

  if (dname.find("Layer_") != string::npos || dname.find("Disk_") != string::npos) {
    vector<string> meVec = iGetter.getMEs();
    for (const auto &it : meVec) {
      string full_path = currDir + "/" + it;
      if (full_path.find("hitmap_siPixelDigis") != string::npos) {  // If we have the hitmap ME
        MonitorElement *me = iGetter.get(full_path);
        if (!me)
          continue;
        string path = full_path;
        while (path.find_last_of('/') != 5)  // Stop before Pixel/
        {
          path = path.substr(0, path.find_last_of('/'));
          //							cout << "\t" <<
          // path
          //<< endl;
          OccupancyMap = iGetter.get(path + "/" + (isbarrel ? "barrel" : "endcap") + "OccupancyMap");

          if (OccupancyMap) {
            for (int i = 1; i != me->getNbinsX() + 1; i++)
              for (int j = 1; j != me->getNbinsY() + 1; j++) {
                float previous = OccupancyMap->getBinContent(i, j);
                OccupancyMap->setBinContent(i, j, previous + me->getBinContent(i, j));
              }
            OccupancyMap->getTH2F()->SetEntries(OccupancyMap->getTH2F()->Integral());
          }
        }
      }
    }
  } else {
    vector<string> subdirs = iGetter.getSubdirs();
    for (const auto &subdir : subdirs) {
      iGetter.cd(subdir);
      iBooker.cd(subdir);
      if (subdir != "Pixel" &&
          ((isbarrel && subdir.find("Barrel") == string::npos) || (!isbarrel && subdir.find("Endcap") == string::npos)))
        continue;
      fillOccupancy(iBooker, iGetter, isbarrel);
      iBooker.goUp();
      iGetter.setCurrentFolder(iBooker.pwd());
    }
  }

  // occupancyprinting cout<<"leaving
  // SiPixelActionExecutor::fillOccupancy..."<<std::endl;
}

//=============================================================================================================

void SiPixelActionExecutor::normaliseAvDigiOcc(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) {
  // occupancyprinting cout<<"entering
  // SiPixelActionExecutor::normaliseAvDigiOcc..."<<std::endl;

  iGetter.cd();

  MonitorElement *roccupancyPlot = iGetter.get("Pixel/averageDigiOccupancy");

  float totalDigisBPIX = 0.;
  float totalDigisFPIX = 0.;
  for (int i = 1; i != 41; i++) {
    if (i < 33)
      totalDigisBPIX += roccupancyPlot->getBinContent(i);
    else
      totalDigisFPIX += roccupancyPlot->getBinContent(i);
  }
  float averageBPIXOcc = totalDigisBPIX / 32.;
  float averageFPIXOcc = totalDigisFPIX / 8.;
  for (int i = 1; i != 41; i++) {
    if (i < 33)
      roccupancyPlot->setBinContent(i, roccupancyPlot->getBinContent(i) / averageBPIXOcc);
    else
      roccupancyPlot->setBinContent(i, roccupancyPlot->getBinContent(i) / averageFPIXOcc);
  }

  iGetter.setCurrentFolder(iBooker.pwd());
}

//=============================================================================================================

void SiPixelActionExecutor::normaliseAvDigiOccVsLumi(DQMStore::IBooker &iBooker,
                                                     DQMStore::IGetter &iGetter,
                                                     int lumisec) {
  iGetter.cd();

  MonitorElement *avgfedDigiOccvsLumi = iGetter.get("Pixel/avgfedDigiOccvsLumi");

  float totalDigisBPIX = 0.;
  float totalDigisFPIX = 0.;
  for (int i = 1; i != 41; i++) {
    if (i < 33)
      totalDigisBPIX += avgfedDigiOccvsLumi->getBinContent(lumisec, i);
    else
      totalDigisFPIX += avgfedDigiOccvsLumi->getBinContent(lumisec, i);
  }
  float averageBPIXOcc = totalDigisBPIX / 32.;
  float averageFPIXOcc = totalDigisFPIX / 8.;
  for (int i = 1; i != 41; i++) {
    if (i < 33)
      avgfedDigiOccvsLumi->setBinContent(lumisec, i, avgfedDigiOccvsLumi->getBinContent(lumisec, i) / averageBPIXOcc);
    else
      avgfedDigiOccvsLumi->setBinContent(lumisec, i, avgfedDigiOccvsLumi->getBinContent(lumisec, i) / averageFPIXOcc);
  }

  iGetter.setCurrentFolder(iBooker.pwd());
}

//=============================================================================================================

void SiPixelActionExecutor::bookEfficiency(DQMStore::IBooker &iBooker, bool isUpgrade) {
  // Barrel
  iBooker.cd();
  iBooker.setCurrentFolder("Pixel/Barrel");
  if (!isUpgrade) {
    if (Tier0Flag_) {
      HitEfficiency_L1 = iBooker.book2D(
          "HitEfficiency_L1", "Hit Efficiency in Barrel_Layer1;Module;Ladder", 9, -4.5, 4.5, 21, -10.5, 10.5);
      HitEfficiency_L2 = iBooker.book2D(
          "HitEfficiency_L2", "Hit Efficiency in Barrel_Layer2;Module;Ladder", 9, -4.5, 4.5, 33, -16.5, 16.5);
      HitEfficiency_L3 = iBooker.book2D(
          "HitEfficiency_L3", "Hit Efficiency in Barrel_Layer3;Module;Ladder", 9, -4.5, 4.5, 45, -22.5, 22.5);
    } else {
      HitEfficiency_L1 = iBooker.book2D(
          "HitEfficiency_L1", "Hit Efficiency in Barrel_Layer1;Module;Ladder", 9, -4.5, 4.5, 21, -10.5, 10.5);
      HitEfficiency_L2 = iBooker.book2D(
          "HitEfficiency_L2", "Hit Efficiency in Barrel_Layer2;Module;Ladder", 9, -4.5, 4.5, 33, -16.5, 16.5);
      HitEfficiency_L3 = iBooker.book2D(
          "HitEfficiency_L3", "Hit Efficiency in Barrel_Layer3;Module;Ladder", 9, -4.5, 4.5, 45, -22.5, 22.5);
    }
  }  // endifNOTUpgrade
  else if (isUpgrade) {
    if (Tier0Flag_) {
      HitEfficiency_L1 =
          iBooker.book2D("HitEfficiency_L1", "Hit Efficiency in Barrel_Layer1;z-side;Ladder", 2, -1., 1., 12, -6., 6.);
      HitEfficiency_L2 = iBooker.book2D(
          "HitEfficiency_L2", "Hit Efficiency in Barrel_Layer2;z-side;Ladder", 2, -1., 1., 28, -14., 14.);
      HitEfficiency_L3 = iBooker.book2D(
          "HitEfficiency_L3", "Hit Efficiency in Barrel_Layer3;z-side;Ladder", 2, -1., 1., 44, -22., 22.);
      HitEfficiency_L4 = iBooker.book2D(
          "HitEfficiency_L4", "Hit Efficiency in Barrel_Layer4;z-side;Ladder", 2, -1., 1., 64, -32., 32.);
    } else {
      HitEfficiency_L1 =
          iBooker.book2D("HitEfficiency_L1", "Hit Efficiency in Barrel_Layer1;Module;Ladder", 8, -4., 4., 12, -6., 6.);
      HitEfficiency_L2 = iBooker.book2D(
          "HitEfficiency_L2", "Hit Efficiency in Barrel_Layer2;Module;Ladder", 8, -4., 4., 28, -14., 14.);
      HitEfficiency_L3 = iBooker.book2D(
          "HitEfficiency_L3", "Hit Efficiency in Barrel_Layer3;Module;Ladder", 8, -4., 4., 44, -22., 22.);
      HitEfficiency_L4 = iBooker.book2D(
          "HitEfficiency_L4", "Hit Efficiency in Barrel_Layer4;Module;Ladder", 8, -4., 4., 64, -32., 32.);
    }
  }  // endifUpgrade
  // Endcap
  iBooker.cd();
  iBooker.setCurrentFolder("Pixel/Endcap");
  if (!isUpgrade) {
    if (Tier0Flag_) {
      HitEfficiency_Dp1 = iBooker.book2D(
          "HitEfficiency_Dp1", "Hit Efficiency in Endcap_Disk_p1;Blade;Panel", 26, -13., 13., 2, 0.5, 2.5);
      HitEfficiency_Dp2 = iBooker.book2D(
          "HitEfficiency_Dp2", "Hit Efficiency in Endcap_Disk_p2;Blade;Panel", 26, -13., 13., 2, 0.5, 2.5);
      HitEfficiency_Dm1 = iBooker.book2D(
          "HitEfficiency_Dm1", "Hit Efficiency in Endcap_Disk_m1;Blade;Panel", 26, -13., 13., 2, 0.5, 2.5);
      HitEfficiency_Dm2 = iBooker.book2D(
          "HitEfficiency_Dm2", "Hit Efficiency in Endcap_Disk_m2;;Blade;Panel", 26, -13., 13., 2, 0.5, 2.5);
    } else {
      HitEfficiency_Dp1 = iBooker.book2D(
          "HitEfficiency_Dp1", "Hit Efficiency in Endcap_Disk_p1;Blades;Modules", 24, -12., 12., 7, 1., 8.);
      HitEfficiency_Dp2 = iBooker.book2D(
          "HitEfficiency_Dp2", "Hit Efficiency in Endcap_Disk_p2;Blades;Modules", 24, -12., 12., 7, 1., 8.);
      HitEfficiency_Dm1 = iBooker.book2D(
          "HitEfficiency_Dm1", "Hit Efficiency in Endcap_Disk_m1;Blades;Modules", 24, -12., 12., 7, 1., 8.);
      HitEfficiency_Dm2 = iBooker.book2D(
          "HitEfficiency_Dm2", "Hit Efficiency in Endcap_Disk_m2;Blades;Modules", 24, -12., 12., 7, 1., 8.);
    }
  } else if (isUpgrade) {
    if (Tier0Flag_) {
      HitEfficiency_Dp1 =
          iBooker.book2D("HitEfficiency_Dp1", "Hit Efficiency in Endcap_Disk_p1;Blades;", 28, -17., 11., 1, 0., 1.);
      HitEfficiency_Dp2 =
          iBooker.book2D("HitEfficiency_Dp2", "Hit Efficiency in Endcap_Disk_p2;Blades;", 28, -17., 11., 1, 0., 1.);
      HitEfficiency_Dp3 =
          iBooker.book2D("HitEfficiency_Dp3", "Hit Efficiency in Endcap_Disk_p3;Blades;", 28, -17., 11., 1, 0., 1.);
      HitEfficiency_Dm1 =
          iBooker.book2D("HitEfficiency_Dm1", "Hit Efficiency in Endcap_Disk_m1;Blades;", 28, -17., 11., 1, 0., 1.);
      HitEfficiency_Dm2 =
          iBooker.book2D("HitEfficiency_Dm2", "Hit Efficiency in Endcap_Disk_m2;Blades;", 28, -17., 11., 1, 0., 1.);
      HitEfficiency_Dm3 =
          iBooker.book2D("HitEfficiency_Dm3", "Hit Efficiency in Endcap_Disk_m3;Blades;", 28, -17., 11., 1, 0., 1.);
    } else {
      HitEfficiency_Dp1 = iBooker.book2D(
          "HitEfficiency_Dp1", "Hit Efficiency in Endcap_Disk_p1;Blades;Modules", 28, -17., 11., 2, 1., 3.);
      HitEfficiency_Dp2 = iBooker.book2D(
          "HitEfficiency_Dp2", "Hit Efficiency in Endcap_Disk_p2;Blades;Modules", 28, -17., 11., 2, 1., 3.);
      HitEfficiency_Dp3 = iBooker.book2D(
          "HitEfficiency_Dp3", "Hit Efficiency in Endcap_Disk_p3;Blades;Modules", 28, -17., 11., 2, 1., 3.);
      HitEfficiency_Dm1 = iBooker.book2D(
          "HitEfficiency_Dm1", "Hit Efficiency in Endcap_Disk_m1;Blades;Modules", 28, -17., 11., 2, 1., 3.);
      HitEfficiency_Dm2 = iBooker.book2D(
          "HitEfficiency_Dm2", "Hit Efficiency in Endcap_Disk_m2;Blades;Modules", 28, -17., 11., 2, 1., 3.);
      HitEfficiency_Dm3 = iBooker.book2D(
          "HitEfficiency_Dm3", "Hit Efficiency in Endcap_Disk_m3;Blades;Modules", 28, -17., 11., 2, 1., 3.);
    }
  }  // endif(isUpgrade)
  iBooker.cd();
  iBooker.cd("Pixel/");
  string bins[] = {"Layer1", "Layer2", "Layer3", "Disk1+", "Disk2+", "Disk1-", "Disk2-"};
  HitEfficiencySummary = iBooker.book1D("HitEfficiencySummary", "Hit efficiency per sub detector", 7, 0, 7);
  HitEfficiencySummary->setAxisTitle("Sub Detector", 1);
  HitEfficiencySummary->setAxisTitle("Efficiency (%)", 2);
  for (int i = 1; i < 8; i++) {
    HitEfficiencySummary->setBinLabel(i, bins[i - 1]);
  }
}

//=============================================================================================================

void SiPixelActionExecutor::createEfficiency(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool isUpgrade) {
  // std::cout<<"entering
  // SiPixelActionExecutor::createEfficiency..."<<std::endl;
  iGetter.cd();
  iBooker.cd();
  fillEfficiency(iBooker, iGetter, true, isUpgrade);  // Barrel
  iGetter.cd();
  iBooker.cd();
  fillEfficiency(iBooker, iGetter, false, isUpgrade);  // Endcap
  iGetter.cd();
  iBooker.cd();
  // std::cout<<"leaving SiPixelActionExecutor::createEfficiency..."<<std::endl;
}

//=============================================================================================================

int SiPixelActionExecutor::getLadder(const std::string &dname_) {
  int biny_ = 0;
  string lad = dname_.substr(dname_.find("Ladder_") + 7, 2);
  if (dname_.find(lad) != string::npos) {
    biny_ = atoi(lad.c_str());
  }
  return biny_;
}

//=============================================================================================================

int SiPixelActionExecutor::getBlade(const std::string &dname_) {
  int binx_ = 0;
  string blad = dname_.substr(dname_.find("Blade_") + 6, 2);
  if (dname_.find(blad) != string::npos) {
    binx_ = atoi(blad.c_str());
  }
  return binx_;
}

//=============================================================================================================

void SiPixelActionExecutor::fillEfficiency(DQMStore::IBooker &iBooker,
                                           DQMStore::IGetter &iGetter,
                                           bool isbarrel,
                                           bool isUpgrade) {
  // cout<<"entering SiPixelActionExecutor::fillEfficiency..."<<std::endl;
  string currDir = iBooker.pwd();
  string dname = currDir.substr(currDir.find_last_of('/') + 1);
  // cout<<"currDir= "<<currDir<< " , dname= "<<dname<<std::endl;

  if (Tier0Flag_) {  // Offline
    if (isbarrel && dname.find("Ladder_") != string::npos) {
      if (!isUpgrade) {
        vector<string> meVec = iGetter.getMEs();
        for (const auto &it : meVec) {
          string full_path = currDir + "/" + it;

          if (full_path.find("missingMod_") != string::npos) {  // If we have missing hits ME

            // Get the MEs that contain missing and valid hits
            MonitorElement *missing = iGetter.get(full_path);
            if (!missing)
              continue;
            string new_path = full_path.replace(full_path.find("missing"), 7, "valid");
            MonitorElement *valid = iGetter.get(new_path);
            if (!valid)
              continue;
            int binx = 0;
            int biny = 0;
            // get the ladder number
            biny = getLadder(dname);  // Current
            if (currDir.find("Shell_mO") != string::npos || currDir.find("Shell_pO") != string::npos) {
              biny = -biny;
            }
            const int nMod = 4;
            for (int i = 1; i < nMod + 1; i++) {
              float hitEfficiency = -1.0;
              float missingHits = 0;
              float validHits = 0;
              binx = i;  // Module
              if (currDir.find("Shell_m") != string::npos)
                binx = -binx;

              missingHits = missing->getBinContent(i);
              validHits = valid->getBinContent(i);
              if (validHits + missingHits > 0.)
                hitEfficiency = validHits / (validHits + missingHits);

              if (currDir.find("Layer_1") != string::npos) {
                HitEfficiency_L1 = iGetter.get("Pixel/Barrel/HitEfficiency_L1");
                if (HitEfficiency_L1)
                  HitEfficiency_L1->setBinContent(HitEfficiency_L1->getTH2F()->FindBin(binx, biny),
                                                  (float)hitEfficiency);
              } else if (currDir.find("Layer_2") != string::npos) {
                HitEfficiency_L2 = iGetter.get("Pixel/Barrel/HitEfficiency_L2");
                if (HitEfficiency_L2)
                  HitEfficiency_L2->setBinContent(HitEfficiency_L2->getTH2F()->FindBin(binx, biny),
                                                  (float)hitEfficiency);
              } else if (currDir.find("Layer_3") != string::npos) {
                HitEfficiency_L3 = iGetter.get("Pixel/Barrel/HitEfficiency_L3");
                if (HitEfficiency_L3)
                  HitEfficiency_L3->setBinContent(HitEfficiency_L3->getTH2F()->FindBin(binx, biny),
                                                  (float)hitEfficiency);
              }
            }
          }
        }
      }  // endifNOTUpgradeInBPix
      else if (isUpgrade) {
        vector<string> meVec = iGetter.getMEs();
        for (const auto &it : meVec) {
          string full_path = currDir + "/" + it;
          if (full_path.find("missing_") != string::npos) {  // If we have missing hits ME
            MonitorElement *me = iGetter.get(full_path);
            if (!me)
              continue;
            float missingHits = me->getEntries();
            string new_path = full_path.replace(full_path.find("missing"), 7, "valid");
            me = iGetter.get(new_path);
            if (!me)
              continue;
            float validHits = me->getEntries();
            float hitEfficiency = -1.;
            if (validHits + missingHits > 0.)
              hitEfficiency = validHits / (validHits + missingHits);
            int binx = 0;
            int biny = 0;
            biny = getLadder(dname);
            if (currDir.find("Shell_mO") != string::npos || currDir.find("Shell_pO") != string::npos) {
              biny = -biny;
            }
            if (currDir.find("Shell_m") != string::npos) {
              binx = 1;
            } else {
              binx = 2;
            }  // x-axis: z-side
            if (currDir.find("Layer_1") != string::npos) {
              HitEfficiency_L1 = iGetter.get("Pixel/Barrel/HitEfficiency_L1");
              if (HitEfficiency_L1)
                HitEfficiency_L1->setBinContent(binx, biny, (float)hitEfficiency);
            } else if (currDir.find("Layer_2") != string::npos) {
              HitEfficiency_L2 = iGetter.get("Pixel/Barrel/HitEfficiency_L2");
              if (HitEfficiency_L2)
                HitEfficiency_L2->setBinContent(binx, biny, (float)hitEfficiency);
            } else if (currDir.find("Layer_3") != string::npos) {
              HitEfficiency_L3 = iGetter.get("Pixel/Barrel/HitEfficiency_L3");
              if (HitEfficiency_L3)
                HitEfficiency_L3->setBinContent(binx, biny, (float)hitEfficiency);
            } else if (currDir.find("Layer_4") != string::npos) {
              HitEfficiency_L4 = iGetter.get("Pixel/Barrel/HitEfficiency_L4");
              if (HitEfficiency_L4)
                HitEfficiency_L4->setBinContent(binx, biny, (float)hitEfficiency);
            }
          }
        }
      }  // endifUpgradeInBPix
    } else if (!isbarrel && dname.find("Blade_") != string::npos && !isUpgrade) {
      vector<string> meVec = iGetter.getMEs();
      for (const auto &it : meVec) {
        string full_path = currDir + "/" + it;
        if (full_path.find("missing_") != string::npos) {  // If we have missing hits ME
          MonitorElement *missing = iGetter.get(full_path);
          if (!missing)
            continue;
          // float missingHits = missing->getEntries();
          string new_path = full_path.replace(full_path.find("missing"), 7, "valid");
          MonitorElement *valid = iGetter.get(new_path);
          if (!valid)
            continue;
          // float validHits = valid->getEntries();
          int binx = 0;
          int biny = 0;
          binx = getBlade(dname);
          if (currDir.find("HalfCylinder_mI") != string::npos || currDir.find("HalfCylinder_pI") != string::npos) {
            binx = binx + 14;
          } else {
            binx = 13 - binx;
          }
          const int nPanel = 2;
          for (int i = 1; i < nPanel + 1; i++) {
            float hitEfficiency = -1.;
            float missingHits = 0;
            float validHits = 0;
            biny = i;
            missingHits = missing->getBinContent(i);
            validHits = valid->getBinContent(i);
            if (validHits + missingHits > 0.)
              hitEfficiency = validHits / (validHits + missingHits);
            if (currDir.find("Disk_1") != string::npos && currDir.find("HalfCylinder_m") != string::npos) {
              HitEfficiency_Dm1 = iGetter.get("Pixel/Endcap/HitEfficiency_Dm1");
              if (HitEfficiency_Dm1)
                HitEfficiency_Dm1->setBinContent(binx, biny, (float)hitEfficiency);
            } else if (currDir.find("Disk_2") != string::npos && currDir.find("HalfCylinder_m") != string::npos) {
              HitEfficiency_Dm2 = iGetter.get("Pixel/Endcap/HitEfficiency_Dm2");
              if (HitEfficiency_Dm2)
                HitEfficiency_Dm2->setBinContent(binx, biny, (float)hitEfficiency);
            } else if (currDir.find("Disk_1") != string::npos && currDir.find("HalfCylinder_p") != string::npos) {
              HitEfficiency_Dp1 = iGetter.get("Pixel/Endcap/HitEfficiency_Dp1");
              if (HitEfficiency_Dp1)
                HitEfficiency_Dp1->setBinContent(binx, biny, (float)hitEfficiency);
            } else if (currDir.find("Disk_2") != string::npos && currDir.find("HalfCylinder_p") != string::npos) {
              HitEfficiency_Dp2 = iGetter.get("Pixel/Endcap/HitEfficiency_Dp2");
              if (HitEfficiency_Dp2)
                HitEfficiency_Dp2->setBinContent(binx, biny, (float)hitEfficiency);
            }
          }  // EndOfFor
        }
      }
    } else if (!isbarrel && dname.find("Blade_") != string::npos && isUpgrade) {
      vector<string> meVec = iGetter.getMEs();
      for (const auto &it : meVec) {
        string full_path = currDir + "/" + it;
        if (full_path.find("missing_") != string::npos) {  // If we have missing hits ME
          MonitorElement *me = iGetter.get(full_path);
          if (!me)
            continue;
          float missingHits = me->getEntries();
          string new_path = full_path.replace(full_path.find("missing"), 7, "valid");
          me = iGetter.get(new_path);
          if (!me)
            continue;
          float validHits = me->getEntries();
          float hitEfficiency = -1.;
          if (validHits + missingHits > 0.)
            hitEfficiency = validHits / (validHits + missingHits);
          int binx = 0;
          int biny = 1;
          binx = getBlade(dname);
          if (currDir.find("HalfCylinder_mI") != string::npos || currDir.find("HalfCylinder_pI") != string::npos) {
            binx = binx + 12;
          } else {
            if (binx == 1)
              binx = 17;
            else if (binx == 2)
              binx = 16;
            else if (binx == 3)
              binx = 15;
            else if (binx == 4)
              binx = 14;
            else if (binx == 5)
              binx = 13;
            else if (binx == 6)
              binx = 12;
            else if (binx == 7)
              binx = 11;
            else if (binx == 8)
              binx = 10;
            else if (binx == 9)
              binx = 9;
            else if (binx == 10)
              binx = 8;
            else if (binx == 11)
              binx = 7;
            else if (binx == 12)
              binx = 6;
            else if (binx == 13)
              binx = 5;
            else if (binx == 14)
              binx = 4;
            else if (binx == 15)
              binx = 3;
            else if (binx == 16)
              binx = 2;
            else if (binx == 17)
              binx = 1;
          }
          if (currDir.find("Disk_1") != string::npos && currDir.find("HalfCylinder_m") != string::npos) {
            HitEfficiency_Dm1 = iGetter.get("Pixel/Endcap/HitEfficiency_Dm1");
            if (HitEfficiency_Dm1)
              HitEfficiency_Dm1->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_2") != string::npos && currDir.find("HalfCylinder_m") != string::npos) {
            HitEfficiency_Dm2 = iGetter.get("Pixel/Endcap/HitEfficiency_Dm2");
            if (HitEfficiency_Dm2)
              HitEfficiency_Dm2->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_3") != string::npos && currDir.find("HalfCylinder_m") != string::npos) {
            HitEfficiency_Dm3 = iGetter.get("Pixel/Endcap/HitEfficiency_Dm3");
            if (HitEfficiency_Dm3)
              HitEfficiency_Dm3->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_1") != string::npos && currDir.find("HalfCylinder_p") != string::npos) {
            HitEfficiency_Dp1 = iGetter.get("Pixel/Endcap/HitEfficiency_Dp1");
            if (HitEfficiency_Dp1)
              HitEfficiency_Dp1->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_2") != string::npos && currDir.find("HalfCylinder_p") != string::npos) {
            HitEfficiency_Dp2 = iGetter.get("Pixel/Endcap/HitEfficiency_Dp2");
            if (HitEfficiency_Dp2)
              HitEfficiency_Dp2->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_3") != string::npos && currDir.find("HalfCylinder_p") != string::npos) {
            HitEfficiency_Dp3 = iGetter.get("Pixel/Endcap/HitEfficiency_Dp3");
            if (HitEfficiency_Dp3)
              HitEfficiency_Dp3->setBinContent(binx, biny, (float)hitEfficiency);
          }
          // std::cout<<"EFFI: "<<currDir<<" , x: "<<binx<<" , y:
          // "<<biny<<std::endl;
        }
      }
    } else {
      vector<string> subdirs = iGetter.getSubdirs();
      for (const auto &subdir : subdirs) {
        iBooker.cd(subdir);
        iGetter.cd(subdir);
        if (subdir != "Pixel" && ((isbarrel && subdir.find("Barrel") == string::npos) ||
                                  (!isbarrel && subdir.find("Endcap") == string::npos)))
          continue;
        fillEfficiency(iBooker, iGetter, isbarrel, isUpgrade);
        iBooker.goUp();
        iGetter.setCurrentFolder(iBooker.pwd());
      }
    }
  } else {  // Online
    if (dname.find("Module_") != string::npos) {
      vector<string> meVec = iGetter.getMEs();
      for (const auto &it : meVec) {
        string full_path = currDir + "/" + it;
        if (full_path.find("missing_") != string::npos) {  // If we have missing hits ME
          MonitorElement *me = iGetter.get(full_path);
          if (!me)
            continue;
          float missingHits = me->getEntries();
          string new_path = full_path.replace(full_path.find("missing"), 7, "valid");
          me = iGetter.get(new_path);
          if (!me)
            continue;
          float validHits = me->getEntries();
          float hitEfficiency = -1.;
          if (validHits + missingHits > 0.)
            hitEfficiency = validHits / (validHits + missingHits);
          int binx = 0;
          int biny = 0;
          if (isbarrel) {
            if (currDir.find("Shell_m") != string::npos) {
              if (currDir.find("Module_4") != string::npos) {
                binx = 1;
              } else if (currDir.find("Module_3") != string::npos) {
                binx = 2;
              }
              if (currDir.find("Module_2") != string::npos) {
                binx = 3;
              } else if (currDir.find("Module_1") != string::npos) {
                binx = 4;
              }
            } else if (currDir.find("Shell_p") != string::npos) {
              if (currDir.find("Module_1") != string::npos) {
                binx = 5;
              } else if (currDir.find("Module_2") != string::npos) {
                binx = 6;
              }
              if (currDir.find("Module_3") != string::npos) {
                binx = 7;
              } else if (currDir.find("Module_4") != string::npos) {
                binx = 8;
              }
            }
            if (!isUpgrade) {
              if (currDir.find("01") != string::npos) {
                biny = 1;
              } else if (currDir.find("02") != string::npos) {
                biny = 2;
              } else if (currDir.find("03") != string::npos) {
                biny = 3;
              } else if (currDir.find("04") != string::npos) {
                biny = 4;
              } else if (currDir.find("05") != string::npos) {
                biny = 5;
              } else if (currDir.find("06") != string::npos) {
                biny = 6;
              } else if (currDir.find("07") != string::npos) {
                biny = 7;
              } else if (currDir.find("08") != string::npos) {
                biny = 8;
              } else if (currDir.find("09") != string::npos) {
                biny = 9;
              } else if (currDir.find("10") != string::npos) {
                biny = 10;
              } else if (currDir.find("11") != string::npos) {
                biny = 11;
              } else if (currDir.find("12") != string::npos) {
                biny = 12;
              } else if (currDir.find("13") != string::npos) {
                biny = 13;
              } else if (currDir.find("14") != string::npos) {
                biny = 14;
              } else if (currDir.find("15") != string::npos) {
                biny = 15;
              } else if (currDir.find("16") != string::npos) {
                biny = 16;
              } else if (currDir.find("17") != string::npos) {
                biny = 17;
              } else if (currDir.find("18") != string::npos) {
                biny = 18;
              } else if (currDir.find("19") != string::npos) {
                biny = 19;
              } else if (currDir.find("20") != string::npos) {
                biny = 20;
              } else if (currDir.find("21") != string::npos) {
                biny = 21;
              } else if (currDir.find("22") != string::npos) {
                biny = 22;
              }
              if (currDir.find("Shell_mO") != string::npos || currDir.find("Shell_pO") != string::npos) {
                if (currDir.find("Layer_1") != string::npos) {
                  biny = biny + 10;
                } else if (currDir.find("Layer_2") != string::npos) {
                  biny = biny + 16;
                } else if (currDir.find("Layer_3") != string::npos) {
                  biny = biny + 22;
                }
              }
            } else if (isUpgrade) {
              if (currDir.find("01") != string::npos) {
                biny = 1;
              } else if (currDir.find("02") != string::npos) {
                biny = 2;
              } else if (currDir.find("03") != string::npos) {
                biny = 3;
              } else if (currDir.find("04") != string::npos) {
                biny = 4;
              } else if (currDir.find("05") != string::npos) {
                biny = 5;
              } else if (currDir.find("06") != string::npos) {
                biny = 6;
              } else if (currDir.find("07") != string::npos) {
                biny = 7;
              } else if (currDir.find("08") != string::npos) {
                biny = 8;
              } else if (currDir.find("09") != string::npos) {
                biny = 9;
              } else if (currDir.find("10") != string::npos) {
                biny = 10;
              } else if (currDir.find("11") != string::npos) {
                biny = 11;
              } else if (currDir.find("12") != string::npos) {
                biny = 12;
              } else if (currDir.find("13") != string::npos) {
                biny = 13;
              } else if (currDir.find("14") != string::npos) {
                biny = 14;
              } else if (currDir.find("15") != string::npos) {
                biny = 15;
              } else if (currDir.find("16") != string::npos) {
                biny = 16;
              } else if (currDir.find("17") != string::npos) {
                biny = 17;
              } else if (currDir.find("18") != string::npos) {
                biny = 18;
              } else if (currDir.find("19") != string::npos) {
                biny = 19;
              } else if (currDir.find("20") != string::npos) {
                biny = 20;
              } else if (currDir.find("21") != string::npos) {
                biny = 21;
              } else if (currDir.find("22") != string::npos) {
                biny = 22;
              } else if (currDir.find("23") != string::npos) {
                biny = 23;
              } else if (currDir.find("24") != string::npos) {
                biny = 24;
              } else if (currDir.find("25") != string::npos) {
                biny = 25;
              } else if (currDir.find("25") != string::npos) {
                biny = 25;
              } else if (currDir.find("26") != string::npos) {
                biny = 26;
              } else if (currDir.find("27") != string::npos) {
                biny = 27;
              } else if (currDir.find("28") != string::npos) {
                biny = 28;
              } else if (currDir.find("29") != string::npos) {
                biny = 29;
              } else if (currDir.find("30") != string::npos) {
                biny = 30;
              } else if (currDir.find("31") != string::npos) {
                biny = 31;
              } else if (currDir.find("32") != string::npos) {
                biny = 32;
              }
              if (currDir.find("Shell_mO") != string::npos || currDir.find("Shell_pO") != string::npos) {
                if (currDir.find("Layer_1") != string::npos) {
                  biny = biny + 6;
                } else if (currDir.find("Layer_2") != string::npos) {
                  biny = biny + 14;
                } else if (currDir.find("Layer_3") != string::npos) {
                  biny = biny + 22;
                } else if (currDir.find("Layer_4") != string::npos) {
                  biny = biny + 32;
                }
              }
            }
          } else {  // endcap
            if (!isUpgrade) {
              if (currDir.find("01") != string::npos) {
                binx = 1;
              } else if (currDir.find("02") != string::npos) {
                binx = 2;
              } else if (currDir.find("03") != string::npos) {
                binx = 3;
              } else if (currDir.find("04") != string::npos) {
                binx = 4;
              } else if (currDir.find("05") != string::npos) {
                binx = 5;
              } else if (currDir.find("06") != string::npos) {
                binx = 6;
              } else if (currDir.find("07") != string::npos) {
                binx = 7;
              } else if (currDir.find("08") != string::npos) {
                binx = 8;
              } else if (currDir.find("09") != string::npos) {
                binx = 9;
              } else if (currDir.find("10") != string::npos) {
                binx = 10;
              } else if (currDir.find("11") != string::npos) {
                binx = 11;
              } else if (currDir.find("12") != string::npos) {
                binx = 12;
              }
              if (currDir.find("HalfCylinder_mO") != string::npos || currDir.find("HalfCylinder_pO") != string::npos) {
                binx = binx + 12;
              }
              if (currDir.find("Panel_1/Module_1") != string::npos) {
                biny = 1;
              } else if (currDir.find("Panel_2/Module_1") != string::npos) {
                biny = 2;
              } else if (currDir.find("Panel_1/Module_2") != string::npos) {
                biny = 3;
              } else if (currDir.find("Panel_2/Module_2") != string::npos) {
                biny = 4;
              } else if (currDir.find("Panel_1/Module_3") != string::npos) {
                biny = 5;
              } else if (currDir.find("Panel_2/Module_3") != string::npos) {
                biny = 6;
              } else if (currDir.find("Panel_1/Module_4") != string::npos) {
                biny = 7;
              }
            } else if (isUpgrade) {
              if (currDir.find("01") != string::npos) {
                binx = 1;
              } else if (currDir.find("02") != string::npos) {
                binx = 2;
              } else if (currDir.find("03") != string::npos) {
                binx = 3;
              } else if (currDir.find("04") != string::npos) {
                binx = 4;
              } else if (currDir.find("05") != string::npos) {
                binx = 5;
              } else if (currDir.find("06") != string::npos) {
                binx = 6;
              } else if (currDir.find("07") != string::npos) {
                binx = 7;
              } else if (currDir.find("08") != string::npos) {
                binx = 8;
              } else if (currDir.find("09") != string::npos) {
                binx = 9;
              } else if (currDir.find("10") != string::npos) {
                binx = 10;
              } else if (currDir.find("11") != string::npos) {
                binx = 11;
              } else if (currDir.find("12") != string::npos) {
                binx = 12;
              } else if (currDir.find("13") != string::npos) {
                binx = 13;
              } else if (currDir.find("14") != string::npos) {
                binx = 14;
              } else if (currDir.find("15") != string::npos) {
                binx = 15;
              } else if (currDir.find("16") != string::npos) {
                binx = 16;
              } else if (currDir.find("17") != string::npos) {
                binx = 17;
              }
              if (currDir.find("HalfCylinder_mO") != string::npos || currDir.find("HalfCylinder_pO") != string::npos) {
                binx = binx + 17;
              }
              if (currDir.find("Panel_1/Module_1") != string::npos) {
                biny = 1;
              } else if (currDir.find("Panel_2/Module_1") != string::npos) {
                biny = 2;
              }
            }  // endif(isUpgrade)
          }

          if (currDir.find("Layer_1") != string::npos) {
            HitEfficiency_L1 = iGetter.get("Pixel/Barrel/HitEfficiency_L1");
            if (HitEfficiency_L1)
              HitEfficiency_L1->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Layer_2") != string::npos) {
            HitEfficiency_L2 = iGetter.get("Pixel/Barrel/HitEfficiency_L2");
            if (HitEfficiency_L2)
              HitEfficiency_L2->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Layer_3") != string::npos) {
            HitEfficiency_L3 = iGetter.get("Pixel/Barrel/HitEfficiency_L3");
            if (HitEfficiency_L3)
              HitEfficiency_L3->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (isUpgrade && (currDir.find("Layer_4") != string::npos)) {
            HitEfficiency_L4 = iGetter.get("Pixel/Barrel/HitEfficiency_L4");
            if (HitEfficiency_L4)
              HitEfficiency_L4->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_1") != string::npos && currDir.find("HalfCylinder_m") != string::npos) {
            HitEfficiency_Dm1 = iGetter.get("Pixel/Endcap/HitEfficiency_Dm1");
            if (HitEfficiency_Dm1)
              HitEfficiency_Dm1->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_2") != string::npos && currDir.find("HalfCylinder_m") != string::npos) {
            HitEfficiency_Dm2 = iGetter.get("Pixel/Endcap/HitEfficiency_Dm2");
            if (HitEfficiency_Dm2)
              HitEfficiency_Dm2->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_3") != string::npos && currDir.find("HalfCylinder_m") != string::npos) {
            HitEfficiency_Dm3 = iGetter.get("Pixel/Endcap/HitEfficiency_Dm3");
            if (HitEfficiency_Dm3)
              HitEfficiency_Dm3->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_1") != string::npos && currDir.find("HalfCylinder_p") != string::npos) {
            HitEfficiency_Dp1 = iGetter.get("Pixel/Endcap/HitEfficiency_Dp1");
            if (HitEfficiency_Dp1)
              HitEfficiency_Dp1->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_2") != string::npos && currDir.find("HalfCylinder_p") != string::npos) {
            HitEfficiency_Dp2 = iGetter.get("Pixel/Endcap/HitEfficiency_Dp2");
            if (HitEfficiency_Dp2)
              HitEfficiency_Dp2->setBinContent(binx, biny, (float)hitEfficiency);
          } else if (currDir.find("Disk_3") != string::npos && currDir.find("HalfCylinder_p") != string::npos) {
            HitEfficiency_Dp3 = iGetter.get("Pixel/Endcap/HitEfficiency_Dp3");
            if (HitEfficiency_Dp3)
              HitEfficiency_Dp3->setBinContent(binx, biny, (float)hitEfficiency);
          }
        }
      }
    } else {
      // cout<<"finding subdirs now"<<std::endl;
      vector<string> subdirs = iGetter.getSubdirs();
      for (const auto &subdir : subdirs) {
        iBooker.cd(subdir);
        iGetter.cd(subdir);
        if (subdir != "Pixel" && ((isbarrel && subdir.find("Barrel") == string::npos) ||
                                  (!isbarrel && subdir.find("Endcap") == string::npos)))
          continue;
        fillEfficiency(iBooker, iGetter, isbarrel, isUpgrade);
        iBooker.goUp();
        iGetter.setCurrentFolder(iBooker.pwd());
      }
    }
  }  // end online/offline
  // cout<<"leaving SiPixelActionExecutor::fillEfficiency..."<<std::endl;
}

//=============================================================================================================

void SiPixelActionExecutor::fillEfficiencySummary(DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter) {
  // cout<<"entering
  // SiPixelActionExecutor::fillEfficiencySummary..."<<std::endl; First we get
  // the summary plot"
  if (!Tier0Flag_)
    return;
  HitEfficiencySummary = iGetter.get("Pixel/HitEfficiencySummary");
  // Now we will loop over the hit efficiency plots and fill it"
  string hitEfficiencyPostfix[] = {"L1", "L2", "L3", "Dp1", "Dp2", "Dm1", "Dm2"};
  std::vector<std::vector<float>> ignoreXBins = {
      {-4, 2}, {4, 4, -1, -3, 3, -4, -3, -2, -1, -4, -3, -2, -1, 1, -4}, {1, -4, 1}, {}, {}, {}, {}};
  std::vector<std::vector<float>> ignoreYBins = {
      {-9, -3}, {1, 16, 1, -13, -13, -5, -5, -5, -5, -6, -6, -6, -6, -8, -8}, {3, 14, 6}, {}, {}, {}, {}};

  for (int i = 0; i < 7; i++) {
    string subdetName = "Endcap/";
    if (i < 3)
      subdetName = "Barrel/";
    char meName[50];
    sprintf(meName, "Pixel/%sHitEfficiency_%s", subdetName.c_str(), hitEfficiencyPostfix[i].c_str());
    MonitorElement *tempHitEffMap = iGetter.get(meName);
    float totalEff = 0.;
    int totalBins = 0;
    TH1 *hitEffMap = tempHitEffMap->getTH1();
    for (int xBin = 1; xBin < tempHitEffMap->getNbinsX() + 1; xBin++) {
      if (fabs(hitEffMap->GetXaxis()->GetBinCenter(xBin)) < 1.)
        continue;
      for (int yBin = 1; yBin < tempHitEffMap->getNbinsY() + 1; yBin++) {
        if (fabs(hitEffMap->GetYaxis()->GetBinCenter(yBin)) < 0.5)
          continue;
        bool ignoreBin = false;
        for (unsigned int j = 0; j < ignoreXBins[i].size(); j++) {
          if (hitEffMap->GetXaxis()->GetBinCenter(xBin) == ignoreXBins[i][j] &&
              hitEffMap->GetYaxis()->GetBinCenter(yBin) == ignoreYBins[i][j]) {
            ignoreBin = true;
            break;
          }
        }
        if (ignoreBin)
          continue;
        if (!(tempHitEffMap->getBinContent(xBin, yBin) < 0.))
          totalEff += tempHitEffMap->getBinContent(xBin, yBin);
        totalBins++;
      }
    }
    float overalEff = 0.;
    //    std::cout << i << " " << totalEff << " " << totalBins << std::endl;
    if (totalBins > 0)
      overalEff = totalEff / float(totalBins);
    HitEfficiencySummary->setBinContent(i + 1, overalEff);
  }
}
