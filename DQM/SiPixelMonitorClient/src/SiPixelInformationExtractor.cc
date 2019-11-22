/*! \file SiPixelInformationExtractor.cc
 *  \brief This class represents ...
 *
 *  (Documentation under development)
 *
 */
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelEDAClient.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "Rtypes.h"
#include "TAxis.h"
#include "TClass.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TImage.h"
#include "TImageDump.h"
#include "TPad.h"
#include "TPaveLabel.h"
#include "TPaveText.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TRandom.h"
#include "TStopwatch.h"
#include "TString.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TText.h"

#include <cmath>
#include <iostream>
#include <map>

#include <cstdlib>  // for free() - Root can allocate with malloc() - sigh...

using namespace std;
using namespace edm;

//------------------------------------------------------------------------------
/*! \brief Constructor of the SiPixelInformationExtractor class.
 *
 */
SiPixelInformationExtractor::SiPixelInformationExtractor(bool offlineXMLfile) : offlineXMLfile_(offlineXMLfile) {
  edm::LogInfo("SiPixelInformationExtractor") << " Creating SiPixelInformationExtractor "
                                              << "\n";

  readReference_ = false;
}

//------------------------------------------------------------------------------
/*! \brief Destructor of the SiPixelInformationExtractor class.
 *
 */
SiPixelInformationExtractor::~SiPixelInformationExtractor() {
  edm::LogInfo("SiPixelInformationExtractor") << " Deleting SiPixelInformationExtractor "
                                              << "\n";
}

//------------------------------------------------------------------------------
/*! \brief Read Configuration File
 *
 */
void SiPixelInformationExtractor::readConfiguration() {}

//============================================================================================================
// --  Return type of ME
//
std::string SiPixelInformationExtractor::getMEType(MonitorElement *theMe) {
  string qtype = theMe->getRootObject()->IsA()->GetName();
  if (qtype.find("TH1") != string::npos) {
    return "TH1";
  } else if (qtype.find("TH2") != string::npos) {
    return "TH2";
  } else if (qtype.find("TH3") != string::npos) {
    return "TH3";
  }
  return "TH1";
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method
 */
void SiPixelInformationExtractor::getItemList(const multimap<string, string> &req_map,
                                              string item_name,
                                              vector<string> &items) {
  items.clear();
  for (multimap<string, string>::const_iterator it = req_map.begin(); it != req_map.end(); it++) {
    if (it->first == item_name) {
      items.push_back(it->second);
    }
  }
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method
 */
bool SiPixelInformationExtractor::hasItem(multimap<string, string> &req_map, string item_name) {
  multimap<string, string>::iterator pos = req_map.find(item_name);
  if (pos != req_map.end())
    return true;
  return false;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method
 */
std::string SiPixelInformationExtractor::getItemValue(const std::multimap<std::string, std::string> &req_map,
                                                      std::string item_name) {
  std::multimap<std::string, std::string>::const_iterator pos = req_map.find(item_name);
  std::string value = " ";
  if (pos != req_map.end()) {
    value = pos->second;
  }
  return value;
}
std::string SiPixelInformationExtractor::getItemValue(std::multimap<std::string, std::string> &req_map,
                                                      std::string item_name) {
  std::multimap<std::string, std::string>::iterator pos = req_map.find(item_name);
  std::string value = " ";
  if (pos != req_map.end()) {
    value = pos->second;
  }
  return value;
}

//
// -- Get color  name from status
//
void SiPixelInformationExtractor::selectColor(string &col, int status) {
  if (status == dqm::qstatus::STATUS_OK)
    col = "#00ff00";
  else if (status == dqm::qstatus::WARNING)
    col = "#ffff00";
  else if (status == dqm::qstatus::ERROR)
    col = "#ff0000";
  else if (status == dqm::qstatus::OTHER)
    col = "#ffa500";
  else
    col = "#0000ff";
}
//
// -- Get Image name from ME
//
void SiPixelInformationExtractor::selectColor(string &col, vector<QReport *> &reports) {
  int istat = 999;
  int status = 0;
  for (vector<QReport *>::const_iterator it = reports.begin(); it != reports.end(); it++) {
    status = (*it)->getStatus();
    if (status > istat)
      istat = status;
  }
  selectColor(col, status);
}
//
// -- Get Image name from status
//
void SiPixelInformationExtractor::selectImage(string &name, int status) {
  if (status == dqm::qstatus::STATUS_OK)
    name = "images/LI_green.gif";
  else if (status == dqm::qstatus::WARNING)
    name = "images/LI_yellow.gif";
  else if (status == dqm::qstatus::ERROR)
    name = "images/LI_red.gif";
  else if (status == dqm::qstatus::OTHER)
    name = "images/LI_orange.gif";
  else
    name = "images/LI_blue.gif";
}
//
// -- Get Image name from ME
//
void SiPixelInformationExtractor::selectImage(string &name, vector<QReport *> &reports) {
  int istat = 999;
  int status = 0;
  for (vector<QReport *>::const_iterator it = reports.begin(); it != reports.end(); it++) {
    status = (*it)->getStatus();
    if (status > istat)
      istat = status;
  }
  selectImage(name, status);
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 */
void SiPixelInformationExtractor::computeStatus(MonitorElement *theME, double &colorValue, pair<double, double> &norm) {
  double normalizationX = 1;
  double normalizationY = 1;
  double meanX = 0;
  double meanY = 0;

  colorValue = 0;

  pair<double, double> normX;
  pair<double, double> normY;

  string theMEType = getMEType(theME);

  if (theMEType.find("TH1") != string::npos) {
    meanX = (double)theME->getMean();
    getNormalization(theME, normX, "TH1");
    normalizationX = fabs(normX.second - normX.first);
    if (normalizationX == 0) {
      normalizationX = 1.E-20;
    }
    colorValue = meanX / normalizationX;
    norm.first = normX.first;
    norm.second = normX.second;
  }

  if (theMEType.find("TH2") != string::npos) {
    meanX = (double)theME->getMean(1);
    meanY = (double)theME->getMean(2);
    getNormalization2D(theME, normX, normY, "TH2");
    normalizationX = fabs(normX.second - normX.first);
    normalizationY = fabs(normY.second - normY.first);
    if (normalizationX == 0) {
      normalizationX = 1.E-20;
    }
    if (normalizationY == 0) {
      normalizationY = 1.E-20;
    }
    double cVX = meanX / normalizationX;
    double cVY = meanY / normalizationY;
    colorValue = sqrt(cVX * cVX + cVY * cVY);
    if (normalizationX >= normalizationY) {
      norm.first = normX.first;
      norm.second = normX.second;
    } else {
      norm.first = normY.first;
      norm.second = normY.second;
    }
  }

  return;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 */
void SiPixelInformationExtractor::getNormalization(MonitorElement *theME,
                                                   pair<double, double> &norm,
                                                   std::string theMEType) {
  double normLow = 0;
  double normHigh = 0;

  if (theMEType.find("TH1") != string::npos) {
    normHigh = (double)theME->getNbinsX();
    norm.first = normLow;
    norm.second = normHigh;
  }
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 */
void SiPixelInformationExtractor::getNormalization2D(MonitorElement *theME,
                                                     pair<double, double> &normX,
                                                     pair<double, double> &normY,
                                                     std::string theMEType) {
  double normLow = 0;
  double normHigh = 0;

  if (theMEType.find("TH2") != string::npos) {
    normHigh = (double)theME->getNbinsX();
    normX.first = normLow;
    normX.second = normHigh;
    normHigh = (double)theME->getNbinsY();
    normY.first = normLow;
    normY.second = normHigh;
  }
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  Given a pointer to ME returns the associated detId
 */
int SiPixelInformationExtractor::getDetId(MonitorElement *mE) {
  const string &mEName = mE->getName();

  int detId = 0;

  if (mEName.find("_3") != string::npos) {
    string detIdString = mEName.substr((mEName.find_last_of("_")) + 1, 9);
    std::istringstream isst;
    isst.str(detIdString);
    isst >> detId;
  }
  return detId;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

void SiPixelInformationExtractor::bookNoisyPixels(DQMStore::IBooker &iBooker, float noiseRate_, bool Tier0Flag) {
  // std::cout<<"BOOK NOISY PIXEL MEs!"<<std::endl;
  iBooker.cd();
  if (noiseRate_ >= 0.) {
    iBooker.setCurrentFolder("Pixel/Barrel");
    EventRateBarrelPixels = iBooker.book1D("barrelEventRate", "Digi event rate for all Barrel pixels", 1000, 0., 0.01);
    EventRateBarrelPixels->setAxisTitle("Event Rate", 1);
    EventRateBarrelPixels->setAxisTitle("Number of Pixels", 2);
    iBooker.cd();
    iBooker.setCurrentFolder("Pixel/Endcap");
    EventRateEndcapPixels = iBooker.book1D("endcapEventRate", "Digi event rate for all Endcap pixels", 1000, 0., 0.01);
    EventRateEndcapPixels->setAxisTitle("Event Rate", 1);
    EventRateEndcapPixels->setAxisTitle("Number of Pixels", 2);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SiPixelInformationExtractor::findNoisyPixels(DQMStore::IBooker &iBooker,
                                                  DQMStore::IGetter &iGetter,
                                                  bool init,
                                                  float noiseRate_,
                                                  int noiseRateDenominator_,
                                                  edm::ESHandle<SiPixelFedCablingMap> theCablingMap) {
  if (init) {
    endOfModules_ = false;
    nevents_ = noiseRateDenominator_;
    if (nevents_ == -1) {
      iBooker.cd();
      iGetter.cd();
      iBooker.setCurrentFolder("Pixel/EventInfo");
      iGetter.setCurrentFolder("Pixel/EventInfo");
      nevents_ = (iGetter.get("Pixel/EventInfo/processedEvents"))->getIntValue();
    }
    iBooker.cd();
    iGetter.cd();
    myfile_.open("NoisyPixelList.txt", ios::app);
    myfile_ << "Noise summary, ran over " << nevents_ << " events, threshold was set to " << noiseRate_ << std::endl;
  }
  string currDir = iBooker.pwd();
  string dname = currDir.substr(currDir.find_last_of("/") + 1);

  if (dname.find("Module_") != string::npos) {
    vector<string> meVec = iGetter.getMEs();
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      string full_path = currDir + "/" + (*it);
      if (full_path.find("hitmap_siPixelDigis") != string::npos) {
        MonitorElement *me = iGetter.get(full_path);
        if (!me)
          continue;
        int detid = getDetId(me);
        int pixcol = -1;
        int pixrow = -1;
        std::vector<std::pair<std::pair<int, int>, float>> noisyPixelsInModule;
        TH2F *hothisto = me->getTH2F();
        if (hothisto) {
          for (int i = 1; i != hothisto->GetNbinsX() + 1; i++) {
            for (int j = 1; j != hothisto->GetNbinsY() + 1; j++) {
              float value = (hothisto->GetBinContent(i, j)) / float(nevents_);
              if (me->getPathname().find("Barrel") != string::npos) {
                EventRateBarrelPixels = iGetter.get("Pixel/Barrel/barrelEventRate");
                if (EventRateBarrelPixels)
                  EventRateBarrelPixels->Fill(value);
              } else if (me->getPathname().find("Endcap") != string::npos) {
                EventRateEndcapPixels = iGetter.get("Pixel/Endcap/endcapEventRate");
                if (EventRateEndcapPixels)
                  EventRateEndcapPixels->Fill(value);
              }
              if (value > noiseRate_) {
                pixcol = i - 1;
                pixrow = j - 1;

                std::pair<int, int> address(pixcol, pixrow);
                std::pair<std::pair<int, int>, float> PixelStats(address, value);
                noisyPixelsInModule.push_back(PixelStats);
              }
            }
          }
        }
        noisyDetIds_[detid] = noisyPixelsInModule;
      }
    }
  }
  vector<string> subDirVec = iGetter.getSubdirs();
  for (vector<string>::const_iterator ic = subDirVec.begin(); ic != subDirVec.end(); ic++) {
    if ((*ic).find("AdditionalPixelErrors") != string::npos)
      continue;
    iGetter.cd(*ic);
    iBooker.cd(*ic);
    init = false;
    findNoisyPixels(iBooker, iGetter, init, noiseRate_, noiseRateDenominator_, theCablingMap);
    iBooker.goUp();
    iGetter.setCurrentFolder(iBooker.pwd());
  }

  if (iBooker.pwd().find("EventInfo") != string::npos)
    endOfModules_ = true;

  if (!endOfModules_)
    return;
  if (currDir == "Pixel/EventInfo/reportSummaryContents") {
    std::vector<std::pair<sipixelobjects::DetectorIndex, double>> pixelvec;
    std::map<uint32_t, int> myfedmap;
    std::map<uint32_t, std::string> mynamemap;
    int realfedID = -1;
    int counter = 0;
    int n_noisyrocs_all = 0;
    int n_noisyrocs_barrel = 0;
    int n_noisyrocs_endcap = 0;
    int n_verynoisyrocs_all = 0;
    int n_verynoisyrocs_barrel = 0;
    int n_verynoisyrocs_endcap = 0;

    for (int fid = 0; fid < 40; fid++) {
      for (std::map<uint32_t, std::vector<std::pair<std::pair<int, int>, float>>>::const_iterator it =
               noisyDetIds_.begin();
           it != noisyDetIds_.end();
           it++) {
        uint32_t detid = (*it).first;
        std::vector<std::pair<std::pair<int, int>, float>> noisyPixels = (*it).second;
        // now convert into online conventions:
        for (int fedid = 0; fedid <= 40; ++fedid) {
          SiPixelFrameConverter converter(theCablingMap.product(), fedid);
          uint32_t newDetId = detid;
          if (converter.hasDetUnit(newDetId)) {
            realfedID = fedid;
            break;
          }
        }
        if (fid == realfedID) {
          if (realfedID == -1)
            continue;
          DetId detId(detid);
          uint32_t detSubId = detId.subdetId();
          std::string outputname;
          bool HalfModule = false;
          if (detSubId == 2) {  // FPIX
            PixelEndcapName nameworker(detid);
            outputname = nameworker.name();
          } else if (detSubId == 1) {  // BPIX
            PixelBarrelName nameworker(detid);
            outputname = nameworker.name();
            HalfModule = nameworker.isHalfModule();

          } else {
            continue;
          }
          std::map<int, int> myrocmap;
          myfedmap[detid] = realfedID;
          mynamemap[detid] = outputname;

          for (std::vector<std::pair<std::pair<int, int>, float>>::const_iterator pxl = noisyPixels.begin();
               pxl != noisyPixels.end();
               pxl++) {
            std::pair<int, int> offlineaddress = (*pxl).first;
            float Noise_frac = (*pxl).second;
            int offlineColumn = offlineaddress.first;
            int offlineRow = offlineaddress.second;
            counter++;

            sipixelobjects::ElectronicIndex cabling;
            SiPixelFrameConverter formatter(theCablingMap.product(), realfedID);
            sipixelobjects::DetectorIndex detector = {detid, offlineRow, offlineColumn};
            formatter.toCabling(cabling, detector);
            // cabling should now contain cabling.roc and cabling.dcol  and
            // cabling.pxid however, the coordinates now need to be converted
            // from dcl,pxid to the row,col coordinates used in the calibration
            // info
            sipixelobjects::LocalPixel::DcolPxid loc;
            loc.dcol = cabling.dcol;
            loc.pxid = cabling.pxid;

            sipixelobjects::LocalPixel locpixel(loc);
            assert(realfedID >= 0);
            assert(cabling.link >= 0);
            assert(cabling.roc >= 0);
            sipixelobjects::CablingPathToDetUnit path = {static_cast<unsigned int>(realfedID),
                                                         static_cast<unsigned int>(cabling.link),
                                                         static_cast<unsigned int>(cabling.roc)};
            const sipixelobjects::PixelROC *theRoc = theCablingMap->findItem(path);
            // END of FIX

            int onlineColumn = locpixel.rocCol();
            int onlineRow = locpixel.rocRow();
            myrocmap[(theRoc->idInDetUnit())]++;

            // ROC numbers in the barrel go from 8 to 15 instead of 0 to 7 in
            // half modules.  This is a fix to get the roc number, and add 8 to
            // it if: it's a Barrel module AND on the minus side AND a Half
            // module

            int rocnumber = -1;

            if ((detSubId == 1) && (outputname.find("mO") != string::npos || outputname.find("mI") != string::npos) &&
                (HalfModule)) {
              rocnumber = theRoc->idInDetUnit() + 8;
            } else {
              rocnumber = theRoc->idInDetUnit();
            }

            myfile_ << "NAME: " << outputname << " , DETID: " << detid << " , OFFLINE: col,row: " << offlineColumn
                    << "," << offlineRow << "  \t , ONLINE: roc,col,row: " << rocnumber << "," << onlineColumn << ","
                    << onlineRow << "  \t , fed,dcol,pixid,link: " << realfedID << "," << loc.dcol << "," << loc.pxid
                    << "," << cabling.link << ", Noise fraction: " << Noise_frac << std::endl;
          }
          for (std::map<int, int>::const_iterator nrc = myrocmap.begin(); nrc != myrocmap.end(); nrc++) {
            if ((*nrc).second > 0) {
              n_noisyrocs_all++;
              if (detSubId == 2) {
                n_noisyrocs_endcap++;
              } else if (detSubId == 1) {
                n_noisyrocs_barrel++;
              }
            }
            if ((*nrc).second > 40) {
              n_verynoisyrocs_all++;
              if (detSubId == 2) {
                n_verynoisyrocs_endcap++;
              } else if (detSubId == 1) {
                n_verynoisyrocs_barrel++;
              }
            }
          }
        }
      }
    }
    myfile_ << "There are " << n_noisyrocs_all
            << " noisy ROCs (ROCs with at least 1 noisy pixel) in the entire "
               "detector. "
            << n_noisyrocs_endcap << " are in the FPIX and " << n_noisyrocs_barrel << " are in the BPIX. " << endl;
    myfile_ << "There are " << n_verynoisyrocs_all
            << " highly noisy ROCs (ROCs with at least 10% of all pixels "
               "passing the noise threshold) in the entire detector. "
            << n_verynoisyrocs_endcap << " are in the FPIX and " << n_verynoisyrocs_barrel << " are in the BPIX. "
            << endl;
  }
  myfile_.close();
  return;
}
