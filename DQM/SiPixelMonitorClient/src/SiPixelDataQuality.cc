/*! \file SiPixelDataQuality.cc
 *  \brief This class represents ...
 *
 *  (Documentation under development)
 *
 */
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelDataQuality.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelEDAClient.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
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
#include <fstream>

using namespace std;
using namespace edm;

//------------------------------------------------------------------------------
/*! \brief Constructor of the SiPixelInformationExtractor class.
 *
 */
SiPixelDataQuality::SiPixelDataQuality(bool offlineXMLfile) : offlineXMLfile_(offlineXMLfile) {
  edm::LogInfo("SiPixelDataQuality") << " Creating SiPixelDataQuality "
                                     << "\n";

  allMods_ = 0;
  errorMods_ = 0;
  qflag_ = 1.;

  allmodsMap = nullptr;
  errmodsMap = nullptr;
  goodmodsMap = nullptr;
  allmodsVec = nullptr;
  errmodsVec = nullptr;
  goodmodsVec = nullptr;
  for (int i = 0; i < 40; ++i) {
    lastallmods_[i] = 0;
    lasterrmods_[i] = 0;
  }
  timeoutCounter_ = 0;
  lastLS_ = -1;
}

//------------------------------------------------------------------------------
/*! \brief Destructor of the SiPixelDataQuality class.
 *
 */
SiPixelDataQuality::~SiPixelDataQuality() {
  edm::LogInfo("SiPixelDataQuality") << " Deleting SiPixelDataQuality "
                                     << "\n";
  if (allmodsMap)
    delete allmodsMap;
  if (errmodsMap)
    delete errmodsMap;
  if (goodmodsMap)
    delete goodmodsMap;
  if (allmodsVec)
    delete allmodsVec;
  if (errmodsVec)
    delete errmodsVec;
  if (goodmodsVec)
    delete goodmodsVec;
}

//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  Given a pointer to ME returns the associated detId
 */
int SiPixelDataQuality::getDetId(MonitorElement *mE) {
  const string &mEName = mE->getName();

  int detId = 0;

  if (mEName.find("_3") != string::npos) {
    string detIdString = mEName.substr((mEName.find_last_of('_')) + 1, 9);
    std::istringstream isst;
    isst.str(detIdString);
    isst >> detId;
  }

  return detId;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

void SiPixelDataQuality::bookGlobalQualityFlag(DQMStore::IBooker &iBooker, bool Tier0Flag, int nFEDs) {
  // std::cout<<"BOOK GLOBAL QUALITY FLAG MEs!"<<std::endl;
  iBooker.cd();

  iBooker.setCurrentFolder("Pixel/Barrel");
  if (!Tier0Flag) {
    ClusterModAll = iBooker.book1D("NClustertoChargeRatio_AllMod", "Cluster Noise All Modules", 768, 0., 768.);
    ClusterMod1 = iBooker.book1D(
        "NClustertoChargeRatio_NormMod1", "Normalized N_{Clusters} to Charge Ratio per Module1", 192, 0., 192.);
    ClusterMod2 = iBooker.book1D(
        "NClustertoChargeRatio_NormMod2", "Normalized N_{Clusters} to Charge Ratio per Module2", 192, 0., 192.);
    ClusterMod3 = iBooker.book1D(
        "NClustertoChargeRatio_NormMod3", "Normalized N_{Clusters} to Charge Ratio per Module3", 192, 0., 192.);
    ClusterMod4 = iBooker.book1D(
        "NClustertoChargeRatio_NormMod4", "Normalized N_{Clusters} to Charge Ratio per Module4", 192, 0., 192.);
  }
  iBooker.setCurrentFolder("Pixel/EventInfo");
  if (!Tier0Flag) {
    SummaryReportMap = iBooker.book2D("reportSummaryMap", "Pixel Summary Map", 3000, 0., 3000., 40, 0., 40.);
    SummaryReportMap->setAxisTitle("Lumi Section", 1);
    SummaryReportMap->setAxisTitle("Pixel FED #", 2);
    allmodsVec = new TH1D("allmodsVec", "allmodsVec", 40, 0., 40.);
    errmodsVec = new TH1D("errmodsVec", "errmodsVec", 40, 0., 40.);
    goodmodsVec = new TH1D("goodmodsVec", "goodmodsVec", 40, 0., 40.);
  } else {
    SummaryReportMap = iBooker.book2D("reportSummaryMap", "Pixel Summary Map", 2, 0., 2., 7, 0., 7.);
    SummaryReportMap->setBinLabel(1, "Barrel", 1);
    SummaryReportMap->setBinLabel(2, "Endcaps", 1);
    SummaryReportMap->setBinLabel(1, "Errors", 2);
    SummaryReportMap->setBinLabel(2, "NDigis", 2);
    SummaryReportMap->setBinLabel(3, "DigiCharge", 2);
    SummaryReportMap->setBinLabel(4, "ClusterSize", 2);
    SummaryReportMap->setBinLabel(5, "NClusters", 2);
    SummaryReportMap->setBinLabel(6, "ClusterCharge", 2);
    SummaryReportMap->setBinLabel(7, "HitEff", 2);
    allmodsMap = new TH2F("allmodsMap", "allmodsMap", 2, 0., 2., 7, 0., 7.);
    errmodsMap = new TH2F("errmodsMap", "errmodsMap", 2, 0., 2., 7, 0., 7.);
    goodmodsMap = new TH2F("goodmodsMap", "goodmodsMap", 2, 0., 2., 7, 0., 7.);
  }
  SummaryPixel = iBooker.bookFloat("reportSummary");
  iBooker.setCurrentFolder("Pixel/EventInfo/reportSummaryContents");
  SummaryBarrel = iBooker.bookFloat("PixelBarrelFraction");
  SummaryEndcap = iBooker.bookFloat("PixelEndcapFraction");
  // book the data certification cuts:
  iBooker.setCurrentFolder("Pixel/AdditionalPixelErrors");
  NErrorsFEDs = iBooker.bookFloat("FEDsNErrorsCut");
  iBooker.setCurrentFolder("Pixel/Barrel");
  NErrorsBarrel = iBooker.bookFloat("BarrelNErrorsCut");
  NDigisBarrel = iBooker.bookInt("BarrelNDigisCut");
  DigiChargeBarrel = iBooker.bookInt("BarrelDigiChargeCut");
  ClusterSizeBarrel = iBooker.bookInt("BarrelClusterSizeCut");
  NClustersBarrel = iBooker.bookInt("BarrelNClustersCut");
  ClusterChargeBarrel = iBooker.bookInt("BarrelClusterChargeCut");
  iBooker.setCurrentFolder("Pixel/Endcap");
  NErrorsEndcap = iBooker.bookFloat("EndcapNErrorsCut");
  NDigisEndcap = iBooker.bookInt("EndcapNDigisCut");
  DigiChargeEndcap = iBooker.bookInt("EndcapDigiChargeCut");
  ClusterSizeEndcap = iBooker.bookInt("EndcapClusterSizeCut");
  NClustersEndcap = iBooker.bookInt("EndcapNClustersCut");
  ClusterChargeEndcap = iBooker.bookInt("EndcapClusterChargeCut");
  if (Tier0Flag) {
    iBooker.setCurrentFolder("Pixel/Tracks");
    NPixelTracks = iBooker.bookInt("PixelTracksCut");
  }

  // Init MonitoringElements:
  if (nFEDs > 0) {
    if (SummaryPixel)
      SummaryPixel->Fill(1.);
    if (SummaryBarrel)
      SummaryBarrel->Fill(1.);
    if (SummaryEndcap)
      SummaryEndcap->Fill(1.);
  } else {
    if (SummaryPixel)
      SummaryPixel->Fill(-1.);
    if (SummaryBarrel)
      SummaryBarrel->Fill(-1.);
    if (SummaryEndcap)
      SummaryEndcap->Fill(-1.);
  }
  if (NErrorsBarrel)
    NErrorsBarrel->Fill(1.);
  if (NErrorsEndcap)
    NErrorsEndcap->Fill(1.);
  if (NErrorsFEDs)
    NErrorsFEDs->Fill(1.);
  if (NDigisBarrel)
    NDigisBarrel->Fill(1);
  if (NDigisEndcap)
    NDigisEndcap->Fill(1);
  if (DigiChargeBarrel)
    DigiChargeBarrel->Fill(1);
  if (DigiChargeEndcap)
    DigiChargeEndcap->Fill(1);
  if (ClusterSizeBarrel)
    ClusterSizeBarrel->Fill(1);
  if (ClusterSizeEndcap)
    ClusterSizeEndcap->Fill(1);
  if (ClusterChargeBarrel)
    ClusterChargeBarrel->Fill(1);
  if (ClusterChargeEndcap)
    ClusterChargeEndcap->Fill(1);
  if (NClustersBarrel)
    NClustersBarrel->Fill(1);
  if (NClustersEndcap)
    NClustersEndcap->Fill(1);
  if (Tier0Flag) {
    if (NPixelTracks)
      NPixelTracks->Fill(1);
  }

  if (SummaryReportMap) {
    if (!Tier0Flag)
      for (int i = 1; i != 3001; i++)
        for (int j = 1; j != 41; j++)
          SummaryReportMap->setBinContent(i, j, -1.);
    if (Tier0Flag)
      for (int i = 1; i != 3; i++)
        for (int j = 1; j != 8; j++)
          SummaryReportMap->setBinContent(i, j, -1.);
  }
  if (!Tier0Flag) {
    for (int j = 1; j != 41; j++) {
      if (allmodsVec)
        allmodsVec->SetBinContent(j, 0.);
      if (errmodsVec)
        errmodsVec->SetBinContent(j, 0.);
      if (goodmodsVec)
        goodmodsVec->SetBinContent(j, 0.);
    }
  }
  if (Tier0Flag) {
    for (int i = 1; i != 3; i++)
      for (int j = 1; j != 8; j++) {
        if (allmodsMap)
          allmodsMap->SetBinContent(i, j, 0.);
        if (errmodsMap)
          errmodsMap->SetBinContent(i, j, 0.);
        if (goodmodsMap)
          goodmodsMap->SetBinContent(i, j, 0.);
      }
  }

  iBooker.cd();
}

//**********************************************************************************************

void SiPixelDataQuality::computeGlobalQualityFlag(
    DQMStore::IBooker &iBooker, DQMStore::IGetter &iGetter, bool init, int nFEDs, bool Tier0Flag) {
  if (init) {
    allMods_ = 0;
    errorMods_ = 0;
    qflag_ = 0.;
    barrelMods_ = 0;
    endcapMods_ = 0;
    objectCount_ = 0;
    DONE_ = false;

    // Error counters and flags:
    n_errors_barrel_ = 0;
    barrel_error_flag_ = 0.;
    n_errors_endcap_ = 0;
    endcap_error_flag_ = 0.;
    n_errors_pixel_ = 0;
    pixel_error_flag_ = 0.;
    digiStatsBarrel = false, clusterStatsBarrel = false, trackStatsBarrel = false;
    digiCounterBarrel = 0, clusterCounterBarrel = 0, trackCounterBarrel = 0;
    digiStatsEndcap = false, clusterStatsEndcap = false, trackStatsEndcap = false;
    digiCounterEndcap = 0, clusterCounterEndcap = 0, trackCounterEndcap = 0;
    init = false;
  }
  if (nFEDs == 0)
    return;

  string currDir = iBooker.pwd();
  string dname = currDir.substr(currDir.find_last_of('/') + 1);

  if ((!Tier0Flag && dname.find("Module_") != string::npos) ||
      (Tier0Flag && (dname.find("Ladder_") != string::npos || dname.find("Blade_") != string::npos))) {
    objectCount_++;

    if (currDir.find("Pixel") != string::npos)
      allMods_++;
    if (currDir.find("Barrel") != string::npos)
      barrelMods_++;
    if (currDir.find("Endcap") != string::npos)
      endcapMods_++;
    vector<string> meVec = iGetter.getMEs();
    for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end(); it++) {
      string full_path = currDir + "/" + (*it);
      if (full_path.find("ndigis_") != string::npos) {
        MonitorElement *me = iGetter.get(full_path);
        if (!me)
          continue;
        if (me->getEntries() > 25) {
          if (full_path.find("Barrel") != string::npos)
            digiCounterBarrel++;
          if (full_path.find("Endcap") != string::npos)
            digiCounterEndcap++;
        }
      } else if (Tier0Flag && full_path.find("nclusters_OnTrack_") != string::npos) {
        MonitorElement *me = iGetter.get(full_path);
        if (!me)
          continue;
        if (me->getEntries() > 25) {
          if (full_path.find("Barrel") != string::npos)
            clusterCounterBarrel++;
          if (full_path.find("Endcap") != string::npos)
            clusterCounterEndcap++;
        }
      } else if (!Tier0Flag && full_path.find("nclusters_") != string::npos) {
        MonitorElement *me = iGetter.get(full_path);
        if (!me)
          continue;
        if (me->getEntries() > 25) {
          if (full_path.find("Barrel") != string::npos)
            clusterCounterBarrel++;
          if (full_path.find("Endcap") != string::npos)
            clusterCounterEndcap++;
        }
      }
    }
  }
  vector<string> subDirVec = iGetter.getSubdirs();
  for (vector<string>::const_iterator ic = subDirVec.begin(); ic != subDirVec.end(); ic++) {
    iGetter.cd(*ic);
    iBooker.cd(*ic);
    init = false;
    computeGlobalQualityFlag(iBooker, iGetter, init, nFEDs, Tier0Flag);
    iBooker.goUp();
    iGetter.setCurrentFolder(iBooker.pwd());
  }

  // Make sure I have finished looping over all Modules/Ladders/Blades:
  if (!Tier0Flag) {  // online case
    if (objectCount_ == 1440)
      DONE_ = true;
  } else {  // offline case
    if (objectCount_ == 288)
      DONE_ = true;
  }

  if (DONE_ && currDir == "Pixel/EventInfo/reportSummaryContents") {
    // Evaluate error flag now, only stored in AdditionalPixelErrors:
    MonitorElement *me_err = iGetter.get("Pixel/AdditionalPixelErrors/FedETypeNErr");
    MonitorElement *me_evt = iGetter.get("Pixel/EventInfo/processedEvents");
    if (me_err && me_evt) {
      for (int i = 1; i != 41; i++)
        for (int j = 1; j != 22; j++)
          if (me_err->getBinContent(i, j) > 0) {
            n_errors_pixel_ = n_errors_pixel_ + int(me_err->getBinContent(i, j));
            if (i < 33)
              n_errors_barrel_ = n_errors_barrel_ + int(me_err->getBinContent(i, j));
            if (i > 32)
              n_errors_endcap_ = n_errors_endcap_ + int(me_err->getBinContent(i, j));
          }
      int NProcEvts = me_evt->getIntValue();
      if (NProcEvts > 0) {
        barrel_error_flag_ = (float(NProcEvts) - float(n_errors_barrel_)) / float(NProcEvts);
        endcap_error_flag_ = (float(NProcEvts) - float(n_errors_endcap_)) / float(NProcEvts);
        pixel_error_flag_ = (float(NProcEvts) - float(n_errors_barrel_) - float(n_errors_endcap_)) / float(NProcEvts);
      }
    }
    NErrorsBarrel = iGetter.get("Pixel/Barrel/BarrelNErrorsCut");
    if (NErrorsBarrel)
      NErrorsBarrel->Fill(barrel_error_flag_);
    NErrorsEndcap = iGetter.get("Pixel/Endcap/EndcapNErrorsCut");
    if (NErrorsEndcap)
      NErrorsEndcap->Fill(endcap_error_flag_);

    string meName0;
    MonitorElement *me;

    // Fill the Digi flags:
    if (!Tier0Flag) {
      meName0 = "Pixel/Barrel/SUMDIG_ndigis_Barrel";
      if (digiCounterBarrel / 768 > 0.5)
        digiStatsBarrel = true;
      if (digiCounterEndcap / 672 > 0.5)
        digiStatsEndcap = true;
    } else {
      meName0 = "Pixel/Barrel/SUMOFF_ndigis_Barrel";
      if (digiCounterBarrel / 192 > 0.5)
        digiStatsBarrel = true;
      if (digiCounterEndcap / 96 > 0.5)
        digiStatsEndcap = true;
    }
    me = iGetter.get(meName0);
    if (me) {
      NDigisBarrel = iGetter.get("Pixel/Barrel/BarrelNDigisCut");
      if (NDigisBarrel && digiStatsBarrel) {
        if (me->hasError())
          NDigisBarrel->Fill(0);
        else
          NDigisBarrel->Fill(1);
      }
    }
    if (!Tier0Flag)
      meName0 = "Pixel/Endcap/SUMDIG_ndigis_Endcap";
    else
      meName0 = "Pixel/Endcap/SUMOFF_ndigis_Endcap";
    me = iGetter.get(meName0);
    if (me) {
      NDigisEndcap = iGetter.get("Pixel/Endcap/EndcapNDigisCut");
      if (NDigisEndcap && digiStatsEndcap) {
        if (me->hasError())
          NDigisEndcap->Fill(0);
        else
          NDigisEndcap->Fill(1);
      }
    }
    if (!Tier0Flag)
      meName0 = "Pixel/Barrel/SUMDIG_adc_Barrel";
    else
      meName0 = "Pixel/Barrel/SUMOFF_adc_Barrel";
    me = iGetter.get(meName0);
    if (me) {
      DigiChargeBarrel = iGetter.get("Pixel/Barrel/BarrelDigiChargeCut");
      if (DigiChargeBarrel && digiStatsBarrel) {
        if (me->hasError())
          DigiChargeBarrel->Fill(0);
        else
          DigiChargeBarrel->Fill(1);
      }
    }
    if (!Tier0Flag)
      meName0 = "Pixel/Endcap/SUMDIG_adc_Endcap";
    else
      meName0 = "Pixel/Endcap/SUMOFF_adc_Endcap";
    me = iGetter.get(meName0);
    if (me) {
      DigiChargeEndcap = iGetter.get("Pixel/Endcap/EndcapDigiChargeCut");
      if (DigiChargeEndcap && digiStatsEndcap) {
        if (me->hasError())
          DigiChargeEndcap->Fill(0);
        else
          DigiChargeEndcap->Fill(1);
      }
    }

    // Fill the Cluster flags:
    if (!Tier0Flag) {
      meName0 = "Pixel/Barrel/SUMCLU_size_Barrel";
      if (clusterCounterBarrel / 768 > 0.5)
        clusterStatsBarrel = true;
      if (clusterCounterEndcap / 672 > 0.5)
        clusterStatsEndcap = true;
    } else {
      meName0 = "Pixel/Barrel/SUMOFF_size_OnTrack_Barrel";
      if (clusterCounterBarrel / 192 > 0.5)
        clusterStatsBarrel = true;
      if (clusterCounterEndcap / 96 > 0.5)
        clusterStatsEndcap = true;
    }
    me = iGetter.get(meName0);
    if (me) {
      ClusterSizeBarrel = iGetter.get("Pixel/Barrel/BarrelClusterSizeCut");
      if (ClusterSizeBarrel && clusterStatsBarrel) {
        if (me->hasError())
          ClusterSizeBarrel->Fill(0);
        else
          ClusterSizeBarrel->Fill(1);
      }
    }
    if (!Tier0Flag)
      meName0 = "Pixel/Endcap/SUMCLU_size_Endcap";
    else
      meName0 = "Pixel/Endcap/SUMOFF_size_OnTrack_Endcap";
    me = iGetter.get(meName0);
    if (me) {
      ClusterSizeEndcap = iGetter.get("Pixel/Endcap/EndcapClusterSizeCut");
      if (ClusterSizeEndcap && clusterStatsEndcap) {
        if (me->hasError())
          ClusterSizeEndcap->Fill(0);
        else
          ClusterSizeEndcap->Fill(1);
      }
    }
    if (!Tier0Flag)
      meName0 = "Pixel/Barrel/SUMCLU_charge_Barrel";
    else
      meName0 = "Pixel/Barrel/SUMOFF_charge_OnTrack_Barrel";
    me = iGetter.get(meName0);
    if (me) {
      ClusterChargeBarrel = iGetter.get("Pixel/Barrel/BarrelClusterChargeCut");
      if (ClusterChargeBarrel && clusterStatsBarrel) {
        if (me->hasError())
          ClusterChargeBarrel->Fill(0);
        else
          ClusterChargeBarrel->Fill(1);
      }
    }
    if (!Tier0Flag)
      meName0 = "Pixel/Endcap/SUMCLU_charge_Endcap";
    else
      meName0 = "Pixel/Endcap/SUMOFF_charge_OnTrack_Endcap";
    me = iGetter.get(meName0);
    if (me) {
      ClusterChargeEndcap = iGetter.get("Pixel/Endcap/EndcapClusterChargeCut");
      if (ClusterChargeEndcap && clusterStatsEndcap) {
        if (me->hasError())
          ClusterChargeEndcap->Fill(0);
        else
          ClusterChargeEndcap->Fill(1);
      }
    }
    if (!Tier0Flag)
      meName0 = "Pixel/Barrel/SUMCLU_nclusters_Barrel";
    else
      meName0 = "Pixel/Barrel/SUMOFF_nclusters_OnTrack_Barrel";
    me = iGetter.get(meName0);
    if (me) {
      NClustersBarrel = iGetter.get("Pixel/Barrel/BarrelNClustersCut");
      if (NClustersBarrel && clusterStatsBarrel) {
        if (me->hasError())
          NClustersBarrel->Fill(0);
        else
          NClustersBarrel->Fill(1);
      }
    }
    if (!Tier0Flag)
      meName0 = "Pixel/Endcap/SUMCLU_nclusters_Endcap";
    else
      meName0 = "Pixel/Endcap/SUMOFF_nclusters_OnTrack_Endcap";
    me = iGetter.get(meName0);
    if (me) {
      NClustersEndcap = iGetter.get("Pixel/Endcap/EndcapNClustersCut");
      if (NClustersEndcap && clusterStatsEndcap) {
        if (me->hasError())
          NClustersEndcap->Fill(0);
        else
          NClustersEndcap->Fill(1);
      }
    }
    // Pixel Track multiplicity / Pixel hit efficiency
    meName0 = "Pixel/Tracks/ntracks_generalTracks";
    me = iGetter.get(meName0);
    if (me) {
      NPixelTracks = iGetter.get("Pixel/Tracks/PixelTracksCut");
      if (NPixelTracks && me->getBinContent(1) > 1000) {
        if ((float)me->getBinContent(2) / (float)me->getBinContent(1) < 0.01) {
          NPixelTracks->Fill(0);
        } else {
          NPixelTracks->Fill(1);
        }
      }
    }

    //********************************************************************************************************

    // Final combination of all Data Quality results:
    float pixelFlag = -1., barrelFlag = -1., endcapFlag = -1.;
    float barrel_errors_temp[1] = {-1.};
    int barrel_cuts_temp[5] = {5 * -1};
    float endcap_errors_temp[1] = {-1.};
    int endcap_cuts_temp[5] = {5 * -1};
    int pixel_cuts_temp[1] = {-1};
    float combinedCuts = 1.;
    int numerator = 0, denominator = 0;

    // Barrel results:
    me = iGetter.get("Pixel/Barrel/BarrelNErrorsCut");
    if (me)
      barrel_errors_temp[0] = me->getFloatValue();
    me = iGetter.get("Pixel/Barrel/BarrelNDigisCut");
    if (me)
      barrel_cuts_temp[0] = me->getIntValue();
    me = iGetter.get("Pixel/Barrel/BarrelDigiChargeCut");
    if (me)
      barrel_cuts_temp[1] = me->getIntValue();
    me = iGetter.get("Pixel/Barrel/BarrelClusterSizeCut");
    if (me)
      barrel_cuts_temp[2] = me->getIntValue();
    me = iGetter.get("Pixel/Barrel/BarrelNClustersCut");
    if (me)
      barrel_cuts_temp[3] = me->getIntValue();
    me = iGetter.get("Pixel/Barrel/BarrelClusterChargeCut");
    if (me)
      barrel_cuts_temp[4] = me->getIntValue();
    for (int k = 0; k != 5; k++) {
      if (barrel_cuts_temp[k] >= 0) {
        numerator = numerator + barrel_cuts_temp[k];
        denominator++;
      }
    }
    if (denominator != 0)
      combinedCuts = float(numerator) / float(denominator);
    barrelFlag = barrel_errors_temp[0] * combinedCuts;

    // Endcap results:
    combinedCuts = 1.;
    numerator = 0;
    denominator = 0;
    me = iGetter.get("Pixel/Endcap/EndcapNErrorsCut");
    if (me)
      endcap_errors_temp[0] = me->getFloatValue();
    me = iGetter.get("Pixel/Endcap/EndcapNDigisCut");
    if (me)
      endcap_cuts_temp[0] = me->getIntValue();
    me = iGetter.get("Pixel/Endcap/EndcapDigiChargeCut");
    if (me)
      endcap_cuts_temp[1] = me->getIntValue();
    me = iGetter.get("Pixel/Endcap/EndcapClusterSizeCut");
    if (me)
      endcap_cuts_temp[2] = me->getIntValue();
    me = iGetter.get("Pixel/Endcap/EndcapNClustersCut");
    if (me)
      endcap_cuts_temp[3] = me->getIntValue();
    me = iGetter.get("Pixel/Endcap/EndcapClusterChargeCut");
    if (me)
      endcap_cuts_temp[4] = me->getIntValue();
    for (int k = 0; k != 5; k++) {
      if (endcap_cuts_temp[k] >= 0) {
        numerator = numerator + endcap_cuts_temp[k];
        denominator++;
      }
    }
    if (denominator != 0)
      combinedCuts = float(numerator) / float(denominator);
    endcapFlag = endcap_errors_temp[0] * combinedCuts;

    // Track results:
    combinedCuts = 1.;
    numerator = 0;
    denominator = 0;
    me = iGetter.get("Pixel/Tracks/PixelTracksCut");
    if (me)
      pixel_cuts_temp[0] = me->getIntValue();

    // Combination of all:
    combinedCuts = 1.;
    numerator = 0;
    denominator = 0;
    for (int k = 0; k != 5; k++) {
      if (barrel_cuts_temp[k] >= 0) {
        numerator = numerator + barrel_cuts_temp[k];
        denominator++;
      }
      if (endcap_cuts_temp[k] >= 0) {
        numerator = numerator + endcap_cuts_temp[k];
        denominator++;
      }
      if (k < 1 && pixel_cuts_temp[k] >= 0) {
        numerator = numerator + pixel_cuts_temp[k];
        denominator++;
      }
    }
    if (denominator != 0)
      combinedCuts = float(numerator) / float(denominator);
    pixelFlag = float(pixel_error_flag_) * float(combinedCuts);

    SummaryPixel = iGetter.get("Pixel/EventInfo/reportSummary");
    if (SummaryPixel)
      SummaryPixel->Fill(pixelFlag);
    SummaryBarrel = iGetter.get("Pixel/EventInfo/reportSummaryContents/PixelBarrelFraction");
    if (SummaryBarrel)
      SummaryBarrel->Fill(barrelFlag);
    SummaryEndcap = iGetter.get("Pixel/EventInfo/reportSummaryContents/PixelEndcapFraction");
    if (SummaryEndcap)
      SummaryEndcap->Fill(endcapFlag);
  }
}

//**********************************************************************************************

void SiPixelDataQuality::computeGlobalQualityFlagByLumi(DQMStore::IGetter &iGetter,
                                                        bool init,
                                                        int nFEDs,
                                                        bool Tier0Flag,
                                                        int nEvents_lastLS_,
                                                        int nErrorsBarrel_lastLS_,
                                                        int nErrorsEndcap_lastLS_) {
  if (nFEDs == 0)
    return;

  // evaluate fatal FED errors for data quality:
  float BarrelRate_LS = 1.;
  float EndcapRate_LS = 1.;
  float PixelRate_LS = 1.;
  MonitorElement *me = iGetter.get("Pixel/AdditionalPixelErrors/byLumiErrors");
  if (me) {
    double nBarrelErrors_LS = me->getBinContent(1) - nErrorsBarrel_lastLS_;
    double nEndcapErrors_LS = me->getBinContent(2) - nErrorsEndcap_lastLS_;
    double nEvents_LS = me->getBinContent(0) - nEvents_lastLS_;
    if (nBarrelErrors_LS / nEvents_LS > 0.5)
      BarrelRate_LS = 0.;
    if (nEndcapErrors_LS / nEvents_LS > 0.5)
      EndcapRate_LS = 0.;
    if ((nBarrelErrors_LS + nEndcapErrors_LS) / nEvents_LS > 0.5)
      PixelRate_LS = 0.;
  }

  // evaluate mean cluster charge on tracks for data quality:
  float BarrelClusterCharge = 1.;
  float EndcapClusterCharge = 1.;
  float PixelClusterCharge = 1.;
  MonitorElement *me1 = iGetter.get("Pixel/Clusters/OnTrack/charge_siPixelClusters_Barrel");
  if (me1 && me1->getMean() < 12.)
    BarrelClusterCharge = 0.;
  MonitorElement *me2 = iGetter.get("Pixel/Clusters/OnTrack/charge_siPixelClusters_Endcap");
  if (me2 && me2->getMean() < 12.)
    EndcapClusterCharge = 0.;
  MonitorElement *me3 = iGetter.get("Pixel/Clusters/OnTrack/charge_siPixelClusters");
  if (me3 && me3->getMean() < 12.)
    PixelClusterCharge = 0.;

  // evaluate average FED occupancy for data quality:
  float BarrelOccupancy = 1.;
  float EndcapOccupancy = 1.;
  float PixelOccupancy = 1.;
  MonitorElement *me4 = iGetter.get("Pixel/averageDigiOccupancy");
  if (me4) {
    double minBarrelOcc = 999999.;
    double maxBarrelOcc = -1.;
    double meanBarrelOcc = 0.;
    double minEndcapOcc = 999999.;
    double maxEndcapOcc = -1.;
    double meanEndcapOcc = 0.;
    for (int i = 1; i != 41; i++) {
      if (i <= 32 && me4->getBinContent(i) < minBarrelOcc)
        minBarrelOcc = me4->getBinContent(i);
      if (i <= 32 && me4->getBinContent(i) > maxBarrelOcc)
        maxBarrelOcc = me4->getBinContent(i);
      if (i <= 32)
        meanBarrelOcc += me4->getBinContent(i);
      if (i > 32 && me4->getBinContent(i) < minEndcapOcc)
        minEndcapOcc = me4->getBinContent(i);
      if (i > 32 && me4->getBinContent(i) > maxEndcapOcc)
        maxEndcapOcc = me4->getBinContent(i);
      if (i > 32)
        meanEndcapOcc += me4->getBinContent(i);
    }
    meanBarrelOcc = meanBarrelOcc / 32.;
    meanEndcapOcc = meanEndcapOcc / 8.;
    if (minBarrelOcc < 0.1 * meanBarrelOcc || maxBarrelOcc > 2.5 * meanBarrelOcc)
      BarrelOccupancy = 0.;
    if (minEndcapOcc < 0.2 * meanEndcapOcc || maxEndcapOcc > 1.8 * meanEndcapOcc)
      EndcapOccupancy = 0.;
    PixelOccupancy = BarrelOccupancy * EndcapOccupancy;
  }

  float pixelFlag = PixelRate_LS * PixelClusterCharge * PixelOccupancy;
  float barrelFlag = BarrelRate_LS * BarrelClusterCharge * BarrelOccupancy;
  float endcapFlag = EndcapRate_LS * EndcapClusterCharge * EndcapOccupancy;
  SummaryPixel = iGetter.get("Pixel/EventInfo/reportSummary");
  if (SummaryPixel)
    SummaryPixel->Fill(pixelFlag);
  SummaryBarrel = iGetter.get("Pixel/EventInfo/reportSummaryContents/PixelBarrelFraction");
  if (SummaryBarrel)
    SummaryBarrel->Fill(barrelFlag);
  SummaryEndcap = iGetter.get("Pixel/EventInfo/reportSummaryContents/PixelEndcapFraction");
  if (SummaryEndcap)
    SummaryEndcap->Fill(endcapFlag);
}

//**********************************************************************************************

void SiPixelDataQuality::fillGlobalQualityPlot(DQMStore::IBooker &iBooker,
                                               DQMStore::IGetter &iGetter,
                                               bool init,
                                               const SiPixelFedCablingMap *theCablingMap,
                                               int nFEDs,
                                               bool Tier0Flag,
                                               int lumisec) {
  // std::cout<<"Entering SiPixelDataQuality::fillGlobalQualityPlot:
  // "<<nFEDs<<std::endl;
  if (init) {
    count = 0;
    errcount = 0;
    init = false;
    count1 = 0;
    count2 = 0;
    count3 = 0;
    count4 = 0;
    count5 = 0;
    count6 = 0;
    modCounter_ = 0;
    if (!Tier0Flag) {
      // The plots that these Vecs are integrated throughout a run
      // So at each lumi section I save their last values (lastmods)
      // And then subtract them out later when filling the SummaryMap
      for (int j = 1; j != 41; j++) {
        if (allmodsVec)
          lastallmods_[j - 1] = allmodsVec->GetBinContent(j);
        if (errmodsVec)
          lasterrmods_[j - 1] = errmodsVec->GetBinContent(j);
        if (allmodsVec)
          allmodsVec->SetBinContent(j, 0.);
        if (errmodsVec)
          errmodsVec->SetBinContent(j, 0.);
        if (goodmodsVec)
          goodmodsVec->SetBinContent(j, 0.);
      }
    }
    if (Tier0Flag) {
      for (int i = 1; i != 3; i++)
        for (int j = 1; j != 8; j++) {
          if (allmodsMap)
            allmodsMap->SetBinContent(i, j, 0.);
          if (errmodsMap)
            errmodsMap->SetBinContent(i, j, 0.);
          if (goodmodsMap)
            goodmodsMap->SetBinContent(i, j, 0.);
        }
    }
  }

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // Fill Maps:
  // Online:
  if (nFEDs == 0)
    return;

  if (!Tier0Flag) {
    // Not elegant, but not sure where else to put this sweet new plot!
    MonitorElement *meTmp = iGetter.get("Pixel/Barrel/NClustertoChargeRatio_AllMod");
    MonitorElement *meTop = iGetter.get("Pixel/Barrel/SUMCLU_nclusters_Barrel");
    MonitorElement *meBot = iGetter.get("Pixel/Barrel/SUMCLU_charge_Barrel");
    if (meTop && meBot && meTmp) {
      for (int bin = 1; bin < 769; ++bin) {
        float content = 0.0;
        if (meBot->getBinContent(bin) > 0.0) {
          content = meTop->getBinContent(bin) / meBot->getBinContent(bin);
        }
        meTmp->setBinContent(bin, content);
      }
      for (int j = 0; j < 4; ++j) {
        static const char buf[] = "Pixel/Barrel/NClustertoChargeRatio_NormMod%i";
        char modplot[sizeof(buf) + 16];
        sprintf(modplot, buf, j + 1);
        MonitorElement *meFinal = iGetter.get(modplot);
        if (!meFinal)
          continue;
        for (int i = 1; i < 769; ++i) {
          int k = 3 - j;
          if (int(i + k) % 4 == 0)
            meFinal->setBinContent(int((i + k) / 4), meTmp->getBinContent(i));
        }
        // Filling done. Now modification.
        float SFLay[3], TotLay[3];
        for (int ll = 0; ll < 3; ++ll)
          TotLay[ll] = 0.0;
        for (int bin = 1; bin < (meFinal->getNbinsX() + 1); ++bin) {
          int layer = int((bin % 48) / 16);
          TotLay[layer] += meFinal->getBinContent(bin);
        }
        float laynorm = TotLay[1] / 64.;
        for (int ll = 0; ll < 3; ++ll) {
          SFLay[ll] = 0.0;
          if (TotLay[ll] > 0.0 && TotLay[1] > 0.0)
            SFLay[ll] = TotLay[1] / TotLay[ll] * (1. / laynorm);
        }
        // now loop through plot
        for (int bin = 1; bin < (meFinal->getNbinsX() + 1); ++bin) {
          // access the layer number for bin: int((i%48)/16)
          int layer = int((bin % 48) / 16);
          float content = meFinal->getBinContent(bin);
          // apply scale factor to bin content
          meFinal->setBinContent(bin, content * SFLay[layer]);
        }
      }
    }

    string currDir = iBooker.pwd();
    if (currDir.find("Reference") != string::npos || currDir.find("Additional") != string::npos)
      return;
    string dname = currDir.substr(currDir.find_last_of('/') + 1);
    if (dname.find("Module_") != string::npos && currDir.find("Reference") == string::npos) {
      vector<string> meVec = iGetter.getMEs();
      int detId = -1;
      int fedId = -1;
      for (vector<string>::const_iterator it = meVec.begin(); it != meVec.end();
           it++) {  // loop over all modules and fill ndigis into allmodsMap
        // checking for any digis or FED errors to decide if this module is in
        // DAQ:
        string full_path = currDir + "/" + (*it);
        if (detId == -1 && full_path.find("SUMOFF") == string::npos &&
            (full_path.find("ndigis") != string::npos && full_path.find("SUMDIG") == string::npos) &&
            (getDetId(iGetter.get(full_path)) > 100)) {
          MonitorElement *me = iGetter.get(full_path);
          if (!me)
            continue;
          if ((full_path.find("ndigis") != string::npos)) {
            modCounter_++;
            detId = getDetId(me);
            for (int fedid = 0; fedid != 40; ++fedid) {
              SiPixelFrameConverter converter(theCablingMap, fedid);
              uint32_t newDetId = detId;
              if (converter.hasDetUnit(newDetId)) {
                fedId = fedid;
                break;
              }
            }
            double NDigis = 0;
            if (full_path.find("ndigis") != string::npos)
              NDigis = me->getEntries();
            float weight = (allmodsVec->GetBinContent(fedId + 1)) + NDigis;
            allmodsVec->SetBinContent(fedId + 1, weight);
          }
        }
      }  // end loop over MEs
    }    // end of module dir's
    vector<string> subDirVec = iGetter.getSubdirs();
    for (vector<string>::const_iterator ic = subDirVec.begin(); ic != subDirVec.end(); ic++) {
      iBooker.cd(*ic);
      iGetter.cd(*ic);
      init = false;
      fillGlobalQualityPlot(iBooker, iGetter, init, theCablingMap, nFEDs, Tier0Flag, lumisec);
      iBooker.goUp();
      iGetter.setCurrentFolder(iBooker.pwd());
    }
    if (modCounter_ == 1440) {
      iBooker.cd("Pixel/EventInfo/reportSummaryContents");
      iGetter.cd("Pixel/EventInfo/reportSummaryContents");
      if (iBooker.pwd() == "Pixel/EventInfo/reportSummaryContents") {
        for (int i = 0; i != 40; i++) {  // loop over FEDs to fetch the errors
          static const char buf[] = "Pixel/AdditionalPixelErrors/FED_%d/FedChNErr";
          char fedplot[sizeof(buf) + 4];
          int NErrors = 0;
          for (int j = 0; j != 37; j++) {  // loop over FED channels within a FED
            sprintf(fedplot, buf, i);
            MonitorElement *me = iGetter.get(fedplot);
            if (me)
              NErrors = NErrors + me->getBinContent(j + 1);
          }
          // If I fill, then I end up majorly overcounting the numbers of
          // errors...
          if (NErrors > 0) {
            errmodsVec->SetBinContent(i + 1, NErrors);
          }
        }
        SummaryReportMap = iGetter.get("Pixel/EventInfo/reportSummaryMap");
        if (SummaryReportMap) {
          float contents = 0.;
          for (int i = 1; i != 41; i++) {
            // Dynamically subtracting previous (integrated) lumi section values
            // in order to only show current lumi section's numbers
            float mydigis = allmodsVec->GetBinContent(i) - lastallmods_[i - 1];
            float myerrs = errmodsVec->GetBinContent(i) - lasterrmods_[i - 1];
            if ((mydigis + myerrs) > 0.) {
              contents = mydigis / (mydigis + myerrs);
            } else {
              // Changed so that dynamic zooming will still
              // advance over these bins(in renderplugins)
              contents = -0.5;
            }
            SummaryReportMap->setBinContent(lumisec + 1, i, contents);
          }  // end for loop over summaryReportMap bins
        }    // end if reportSummaryMap ME exists
      }      // end if in summary directory
    }        // end if modCounter_
  } else {   // Offline
    float barrel_errors_temp[1] = {-1.};
    int barrel_cuts_temp[6] = {6 * -1};
    float endcap_errors_temp[1] = {-1.};
    int endcap_cuts_temp[6] = {6 * -1};
    int pixel_cuts_temp[1] = {-1};
    // Barrel results:
    MonitorElement *me;
    me = iGetter.get("Pixel/Barrel/BarrelNErrorsCut");
    if (me)
      barrel_errors_temp[0] = me->getFloatValue();
    me = iGetter.get("Pixel/Endcap/EndcapNErrorsCut");
    if (me)
      endcap_errors_temp[0] = me->getFloatValue();
    SummaryReportMap->setBinContent(1, 1, barrel_errors_temp[0]);
    SummaryReportMap->setBinContent(2, 1, endcap_errors_temp[0]);
    me = iGetter.get("Pixel/Barrel/BarrelNDigisCut");
    if (me)
      barrel_cuts_temp[0] = me->getIntValue();
    me = iGetter.get("Pixel/Barrel/BarrelDigiChargeCut");
    if (me)
      barrel_cuts_temp[1] = me->getIntValue();
    me = iGetter.get("Pixel/Barrel/BarrelClusterSizeCut");
    if (me)
      barrel_cuts_temp[2] = me->getIntValue();
    me = iGetter.get("Pixel/Barrel/BarrelNClustersCut");
    if (me)
      barrel_cuts_temp[3] = me->getIntValue();
    me = iGetter.get("Pixel/Barrel/BarrelClusterChargeCut");
    if (me)
      barrel_cuts_temp[4] = me->getIntValue();
    me = iGetter.get("Pixel/Endcap/EndcapNDigisCut");
    if (me)
      endcap_cuts_temp[0] = me->getIntValue();
    me = iGetter.get("Pixel/Endcap/EndcapDigiChargeCut");
    if (me)
      endcap_cuts_temp[1] = me->getIntValue();
    me = iGetter.get("Pixel/Endcap/EndcapClusterSizeCut");
    if (me)
      endcap_cuts_temp[2] = me->getIntValue();
    me = iGetter.get("Pixel/Endcap/EndcapNClustersCut");
    if (me)
      endcap_cuts_temp[3] = me->getIntValue();
    me = iGetter.get("Pixel/Endcap/EndcapClusterChargeCut");
    if (me)
      endcap_cuts_temp[4] = me->getIntValue();
    for (int j = 2; j != 7; j++) {
      SummaryReportMap->setBinContent(1, j, barrel_cuts_temp[j - 2]);
      SummaryReportMap->setBinContent(2, j, endcap_cuts_temp[j - 2]);
    }
    me = iGetter.get("Pixel/Tracks/PixelTracksCut");
    if (me)
      pixel_cuts_temp[0] = me->getIntValue();
    SummaryReportMap->setBinContent(1, 7, pixel_cuts_temp[0]);
    SummaryReportMap->setBinContent(2, 7, pixel_cuts_temp[0]);
  }  // end of offline map
  if (allmodsMap)
    allmodsMap->Clear();
  if (goodmodsMap)
    goodmodsMap->Clear();
  if (errmodsMap)
    errmodsMap->Clear();
}

//**********************************************************************************************
