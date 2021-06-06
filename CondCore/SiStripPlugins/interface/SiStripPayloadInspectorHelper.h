#ifndef CONDCORE_SISTRIPPLUGINS_SISTRIPPAYLOADINSPECTORHELPER_H
#define CONDCORE_SISTRIPPLUGINS_SISTRIPPAYLOADINSPECTORHELPER_H

#include <vector>
#include <numeric>
#include <string>
#include "TH1.h"
#include "TH2.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripDetSummary.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace SiStripPI {

  //##### for plotting

  enum OpMode { STRIP_BASED, APV_BASED, MODULE_BASED };

  class Entry {
  public:
    Entry() : entries(0), sum(0), sq_sum(0) {}

    double mean() { return sum / entries; }
    double std_dev() {
      double tmean = mean();
      return sqrt((sq_sum - entries * tmean * tmean) / (entries - 1));
    }
    double mean_rms() { return std_dev() / sqrt(entries); }

    void add(double val) {
      entries++;
      sum += val;
      sq_sum += val * val;
    }

    void reset() {
      entries = 0;
      sum = 0;
      sq_sum = 0;
    }

  private:
    long int entries;
    double sum, sq_sum;
  };

  // class Monitor1D

  class Monitor1D {
  public:
    Monitor1D(OpMode mode, const char* name, const char* title, int nbinsx, double xmin, double xmax)
        : entry_(), mode_(mode), obj_(name, title, nbinsx, xmin, xmax) {}

    Monitor1D() : entry_(), mode_(OpMode::STRIP_BASED), obj_() {}

    ~Monitor1D() {}

    void Fill(int apv, int det, double vx) {
      switch (mode_) {
        case (OpMode::APV_BASED):
          if (!((apv == prev_apv_ && det == prev_det_) || prev_apv_ == 0)) {
            flush();
          }
          prev_apv_ = apv;
          prev_det_ = det;
          break;
        case (OpMode::MODULE_BASED):
          if (!(det == prev_det_ || prev_det_ == 0)) {
            flush();
          }
          prev_det_ = det;
          break;
        case (OpMode::STRIP_BASED):
          flush();
          break;
      }
      entry_.add(vx);
    }

    void flush() {
      obj_.Fill(entry_.mean());
      entry_.reset();
    }

    TH1F& hist() {
      flush();
      return obj_;
    }

    TH1F& getHist() { return obj_; }

  private:
    int prev_apv_ = 0, prev_det_ = 0;
    Entry entry_;
    OpMode mode_;
    TH1F obj_;
  };

  // class monitor 2D

  class Monitor2D {
  public:
    Monitor2D(OpMode mode,
              const char* name,
              const char* title,
              int nbinsx,
              double xmin,
              double xmax,
              int nbinsy,
              double ymin,
              double ymax)
        : entryx_(), entryy_(), mode_(mode), obj_(name, title, nbinsx, xmin, xmax, nbinsy, ymin, ymax) {}

    Monitor2D() : entryx_(), entryy_(), mode_(OpMode::STRIP_BASED), obj_() {}

    ~Monitor2D() {}

    void Fill(int apv, int det, double vx, double vy) {
      switch (mode_) {
        case (OpMode::APV_BASED):
          if (!((apv == prev_apv_ && det == prev_det_) || prev_apv_ == 0)) {
            flush();
          }
          prev_apv_ = apv;
          prev_det_ = det;
          break;
        case (OpMode::MODULE_BASED):
          if (!(det == prev_det_ || prev_det_ == 0)) {
            flush();
          }
          prev_det_ = det;
          break;
        case (OpMode::STRIP_BASED):
          flush();
          break;
      }
      entryx_.add(vx);
      entryy_.add(vy);
    }

    void flush() {
      obj_.Fill(entryx_.mean(), entryy_.mean());
      entryx_.reset();
      entryy_.reset();
    }

    TH2F& hist() {
      flush();
      return obj_;
    }

  private:
    int prev_apv_ = 0, prev_det_ = 0;
    Entry entryx_, entryy_;
    OpMode mode_;
    TH2F obj_;
  };

  enum estimator { min, max, mean, rms };

  /*--------------------------------------------------------------------*/
  std::string estimatorType(SiStripPI::estimator e)
  /*--------------------------------------------------------------------*/
  {
    switch (e) {
      case SiStripPI::min:
        return "minimum";
      case SiStripPI::max:
        return "maximum";
      case SiStripPI::mean:
        return "mean";
      case SiStripPI::rms:
        return "RMS";
      default:
        return "should never be here";
    }
  }

  /*--------------------------------------------------------------------*/
  std::string getStringFromSubdet(StripSubdetector::SubDetector sub)
  /*-------------------------------------------------------------------*/
  {
    switch (sub) {
      case StripSubdetector::TIB:
        return "TIB";
      case StripSubdetector::TOB:
        return "TOB";
      case StripSubdetector::TID:
        return "TID";
      case StripSubdetector::TEC:
        return "TEC";
      default:
        return "should never be here";
    }
  }

  enum TrackerRegion {
    TIB1r = 1010,
    TIB1s = 1011,
    TIB2r = 1020,
    TIB2s = 1021,
    TIB3r = 1030,
    TIB4r = 1040,
    TOB1r = 2010,
    TOB1s = 2011,
    TOB2r = 2020,
    TOB2s = 2021,
    TOB3r = 2030,
    TOB4r = 2040,
    TOB5r = 2050,
    TOB6r = 2060,
    TEC1r = 3010,
    TEC1s = 3011,
    TEC2r = 3020,
    TEC2s = 3021,
    TEC3r = 3030,
    TEC3s = 3031,
    TEC4r = 3040,
    TEC4s = 3041,
    TEC5r = 3050,
    TEC5s = 3051,
    TEC6r = 3060,
    TEC6s = 3061,
    TEC7r = 3070,
    TEC7s = 3071,
    TEC8r = 3080,
    TEC8s = 3081,
    TEC9r = 3090,
    TEC9s = 3091,
    TID1r = 4010,
    TID1s = 4011,
    TID2r = 4020,
    TID2s = 4021,
    TID3r = 4030,
    TID3s = 4031,
    END_OF_REGIONS
  };

  /*--------------------------------------------------------------------*/
  std::pair<int, const char*> regionType(int index)
  /*--------------------------------------------------------------------*/
  {
    auto region = static_cast<std::underlying_type_t<SiStripPI::TrackerRegion>>(index);

    switch (region) {
      case SiStripPI::TIB1r:
        return std::make_pair(1, "TIB L1 r-#varphi");
      case SiStripPI::TIB1s:
        return std::make_pair(2, "TIB L1 stereo");
      case SiStripPI::TIB2r:
        return std::make_pair(3, "TIB L2 r-#varphi");
      case SiStripPI::TIB2s:
        return std::make_pair(4, "TIB L2 stereo");
      case SiStripPI::TIB3r:
        return std::make_pair(5, "TIB L3");
      case SiStripPI::TIB4r:
        return std::make_pair(6, "TIB L4");
      case SiStripPI::TOB1r:
        return std::make_pair(7, "TOB L1 r-#varphi");
      case SiStripPI::TOB1s:
        return std::make_pair(8, "TOB L1 stereo");
      case SiStripPI::TOB2r:
        return std::make_pair(9, "TOB L2 r-#varphi");
      case SiStripPI::TOB2s:
        return std::make_pair(10, "TOB L2 stereo");
      case SiStripPI::TOB3r:
        return std::make_pair(11, "TOB L3 r-#varphi");
      case SiStripPI::TOB4r:
        return std::make_pair(12, "TOB L4");
      case SiStripPI::TOB5r:
        return std::make_pair(13, "TOB L5");
      case SiStripPI::TOB6r:
        return std::make_pair(14, "TOB L6");
      case SiStripPI::TEC1r:
        return std::make_pair(15, "TEC D1 r-#varphi");
      case SiStripPI::TEC1s:
        return std::make_pair(16, "TEC D1 stereo");
      case SiStripPI::TEC2r:
        return std::make_pair(17, "TEC D2 r-#varphi");
      case SiStripPI::TEC2s:
        return std::make_pair(18, "TEC D2 stereo");
      case SiStripPI::TEC3r:
        return std::make_pair(19, "TEC D3 r-#varphi");
      case SiStripPI::TEC3s:
        return std::make_pair(20, "TEC D3 stereo");
      case SiStripPI::TEC4r:
        return std::make_pair(21, "TEC D4 r-#varphi");
      case SiStripPI::TEC4s:
        return std::make_pair(22, "TEC D4 stereo");
      case SiStripPI::TEC5r:
        return std::make_pair(23, "TEC D5 r-#varphi");
      case SiStripPI::TEC5s:
        return std::make_pair(24, "TEC D5 stereo");
      case SiStripPI::TEC6r:
        return std::make_pair(25, "TEC D6 r-#varphi");
      case SiStripPI::TEC6s:
        return std::make_pair(26, "TEC D6 stereo");
      case SiStripPI::TEC7r:
        return std::make_pair(27, "TEC D7 r-#varphi");
      case SiStripPI::TEC7s:
        return std::make_pair(28, "TEC D7 stereo");
      case SiStripPI::TEC8r:
        return std::make_pair(29, "TEC D8 r-#varphi");
      case SiStripPI::TEC8s:
        return std::make_pair(30, "TEC D8 stereo");
      case SiStripPI::TEC9r:
        return std::make_pair(31, "TEC D9 r-#varphi");
      case SiStripPI::TEC9s:
        return std::make_pair(32, "TEC D9 stereo");
      case SiStripPI::TID1r:
        return std::make_pair(33, "TID D1 r-#varphi");
      case SiStripPI::TID1s:
        return std::make_pair(34, "TID D1 stereo");
      case SiStripPI::TID2r:
        return std::make_pair(35, "TID D2 r-#varphi");
      case SiStripPI::TID2s:
        return std::make_pair(36, "TID D2 stereo");
      case SiStripPI::TID3r:
        return std::make_pair(37, "TID D3 r-#varphi");
      case SiStripPI::TID3s:
        return std::make_pair(38, "TID D3 stereo");
      case SiStripPI::END_OF_REGIONS:
        return std::make_pair(-1, "undefined");
      default:
        return std::make_pair(999, "should never be here");
    }
  }

  /*--------------------------------------------------------------------*/
  std::pair<float, float> getTheRange(std::map<uint32_t, float> values, const float nsigma)
  /*--------------------------------------------------------------------*/
  {
    float sum = std::accumulate(
        std::begin(values), std::end(values), 0.0, [](float value, const std::map<uint32_t, float>::value_type& p) {
          return value + p.second;
        });

    float m = sum / values.size();

    float accum = 0.0;
    std::for_each(std::begin(values), std::end(values), [&](const std::map<uint32_t, float>::value_type& p) {
      accum += (p.second - m) * (p.second - m);
    });

    float stdev = sqrt(accum / (values.size() - 1));

    if (stdev != 0.) {
      return std::make_pair(m - nsigma * stdev, m + nsigma * stdev);
    } else {
      return std::make_pair(m > 0. ? 0.95 * m : 1.05 * m, m > 0 ? 1.05 * m : 0.95 * m);
    }
  }

  /*--------------------------------------------------------------------*/
  void drawStatBox(std::map<std::string, std::shared_ptr<TH1F>> histos,
                   std::map<std::string, int> colormap,
                   std::vector<std::string> legend,
                   double X = 0.15,
                   double Y = 0.93,
                   double W = 0.15,
                   double H = 0.10)
  /*--------------------------------------------------------------------*/
  {
    char buffer[255];

    int i = 0;
    for (const auto& element : legend) {
      TPaveText* stat = new TPaveText(X, Y - (i * H), X + W, Y - (i + 1) * H, "NDC");
      i++;
      auto Histo = histos[element];
      sprintf(buffer, "Entries : %i\n", (int)Histo->GetEntries());
      stat->AddText(buffer);

      sprintf(buffer, "Mean    : %6.2f\n", Histo->GetMean());
      stat->AddText(buffer);

      sprintf(buffer, "RMS     : %6.2f\n", Histo->GetRMS());
      stat->AddText(buffer);

      stat->SetFillColor(0);
      stat->SetLineColor(colormap[element]);
      stat->SetTextColor(colormap[element]);
      stat->SetTextSize(0.03);
      stat->SetBorderSize(0);
      stat->SetMargin(0.05);
      stat->SetTextAlign(12);
      stat->Draw();
    }
  }

  /*--------------------------------------------------------------------*/
  std::pair<float, float> getExtrema(TH1* h1, TH1* h2)
  /*--------------------------------------------------------------------*/
  {
    float theMax(-9999.);
    float theMin(9999.);
    theMax = h1->GetMaximum() > h2->GetMaximum() ? h1->GetMaximum() : h2->GetMaximum();
    theMin = h1->GetMinimum() < h2->GetMaximum() ? h1->GetMinimum() : h2->GetMinimum();

    float add_min = theMin > 0. ? -0.05 : 0.05;
    float add_max = theMax > 0. ? 0.05 : -0.05;

    auto result = std::make_pair(theMin * (1 + add_min), theMax * (1 + add_max));
    return result;
  }

  /*--------------------------------------------------------------------*/
  void makeNicePlotStyle(TH1* hist)
  /*--------------------------------------------------------------------*/
  {
    hist->SetStats(kFALSE);
    hist->SetLineWidth(2);
    hist->GetXaxis()->CenterTitle(true);
    hist->GetYaxis()->CenterTitle(true);
    hist->GetXaxis()->SetTitleFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleSize(0.05);
    hist->GetYaxis()->SetTitleSize(0.05);
    hist->GetXaxis()->SetTitleOffset(0.9);
    hist->GetYaxis()->SetTitleOffset(1.3);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelSize(.05);
    hist->GetXaxis()->SetLabelSize(.05);
  }

  /*--------------------------------------------------------------------*/
  void printSummary(const std::map<unsigned int, SiStripDetSummary::Values>& map)
  /*--------------------------------------------------------------------*/
  {
    for (const auto& element : map) {
      int count = element.second.count;
      double mean = count > 0 ? (element.second.mean) / count : 0.;
      double rms = count > 0 ? (element.second.rms) / count - mean * mean : 0.;
      if (rms <= 0)
        rms = 0;
      else
        rms = sqrt(rms);

      std::string detector;

      switch ((element.first) / 1000) {
        case 1:
          detector = "TIB ";
          break;
        case 2:
          detector = "TOB ";
          break;
        case 3:
          detector = "TEC ";
          break;
        case 4:
          detector = "TID ";
          break;
      }

      int layer = (element.first) / 10 - (element.first) / 1000 * 100;
      int stereo = (element.first) - (layer * 10) - (element.first) / 1000 * 1000;

      std::cout << "key of the map:" << element.first << " ( region: " << regionType(element.first).second << " ) "
                << detector << " layer: " << layer << " stereo:" << stereo << "| count:" << count << " mean: " << mean
                << " rms: " << rms << std::endl;
    }
  }

  // code is mutuated from CalibTracker/SiStripQuality/plugins/SiStripQualityStatistics

  /*--------------------------------------------------------------------*/
  void setBadComponents(int i, int component, const SiStripQuality::BadComponent& BC, int NBadComponent[4][19][4])
  /*--------------------------------------------------------------------*/
  {
    if (BC.BadApvs) {
      NBadComponent[i][0][2] += std::bitset<16>(BC.BadApvs & 0x3f).count();
      NBadComponent[i][component][2] += std::bitset<16>(BC.BadApvs & 0x3f).count();
    }

    if (BC.BadFibers) {
      NBadComponent[i][0][1] += std::bitset<4>(BC.BadFibers & 0x7).count();
      NBadComponent[i][component][1] += std::bitset<4>(BC.BadFibers & 0x7).count();
    }

    if (BC.BadModule) {
      NBadComponent[i][0][0]++;
      NBadComponent[i][component][0]++;
    }
  }

  // generic code to fill a SiStripDetSummary with Noise payload info
  /*--------------------------------------------------------------------*/
  void fillNoiseDetSummary(SiStripDetSummary& summaryNoise,
                           std::shared_ptr<SiStripNoises> payload,
                           SiStripPI::estimator est)
  /*--------------------------------------------------------------------*/
  {
    SiStripNoises::RegistryIterator rit = payload->getRegistryVectorBegin(), erit = payload->getRegistryVectorEnd();
    uint16_t Nstrips;
    std::vector<float> vstripnoise;
    double mean, rms, min, max;
    for (; rit != erit; ++rit) {
      Nstrips = (rit->iend - rit->ibegin) * 8 / 9;  //number of strips = number of chars * char size / strip noise size
      vstripnoise.resize(Nstrips);
      payload->allNoises(
          vstripnoise,
          make_pair(payload->getDataVectorBegin() + rit->ibegin, payload->getDataVectorBegin() + rit->iend));

      mean = 0;
      rms = 0;
      min = 10000;
      max = 0;

      DetId detId(rit->detid);

      for (size_t i = 0; i < Nstrips; ++i) {
        mean += vstripnoise[i];
        rms += vstripnoise[i] * vstripnoise[i];
        if (vstripnoise[i] < min)
          min = vstripnoise[i];
        if (vstripnoise[i] > max)
          max = vstripnoise[i];
      }

      mean /= Nstrips;
      if ((rms / Nstrips - mean * mean) > 0.) {
        rms = sqrt(rms / Nstrips - mean * mean);
      } else {
        rms = 0.;
      }

      switch (est) {
        case SiStripPI::min:
          summaryNoise.add(detId, min);
          break;
        case SiStripPI::max:
          summaryNoise.add(detId, max);
          break;
        case SiStripPI::mean:
          summaryNoise.add(detId, mean);
          break;
        case SiStripPI::rms:
          summaryNoise.add(detId, rms);
          break;
        default:
          edm::LogWarning("LogicError") << "Unknown estimator: " << est;
          break;
      }
    }
  }

  /*--------------------------------------------------------------------*/
  void fillTotalComponents(int NTkComponents[4], int NComponents[4][19][4], const TrackerTopology m_trackerTopo)
  /*--------------------------------------------------------------------*/
  {
    edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
    SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());
    const std::map<uint32_t, SiStripDetInfoFileReader::DetInfo>& DetInfos = reader->getAllData();
    for (const auto& det : DetInfos) {
      int nAPVs = reader->getNumberOfApvsAndStripLength(det.first).first;
      // one fiber connects to 2 APVs
      int nFibers = nAPVs / 2;
      int nStrips = (128 * reader->getNumberOfApvsAndStripLength(det.first).first);
      NTkComponents[0]++;
      NTkComponents[1] += nFibers;
      NTkComponents[2] += nAPVs;
      NTkComponents[3] += nStrips;

      DetId detectorId = DetId(det.first);
      int subDet = detectorId.subdetId();

      int subDetIndex = -1;
      int component = -1;
      if (subDet == StripSubdetector::TIB) {
        subDetIndex = 0;
        component = m_trackerTopo.tibLayer(det.first);
      } else if (subDet == StripSubdetector::TID) {
        subDetIndex = 1;
        component = m_trackerTopo.tidSide(det.first) == 2 ? m_trackerTopo.tidWheel(det.first)
                                                          : m_trackerTopo.tidWheel(det.first) + 3;
      } else if (subDet == StripSubdetector::TOB) {
        subDetIndex = 2;
        component = m_trackerTopo.tobLayer(det.first);
      } else if (subDet == StripSubdetector::TEC) {
        subDetIndex = 3;
        component = m_trackerTopo.tecSide(det.first) == 2 ? m_trackerTopo.tecWheel(det.first)
                                                          : m_trackerTopo.tecWheel(det.first) + 9;
      }

      NComponents[subDetIndex][0][0]++;
      NComponents[subDetIndex][0][1] += nFibers;
      NComponents[subDetIndex][0][2] += nAPVs;
      NComponents[subDetIndex][0][3] += nStrips;

      NComponents[subDetIndex][component][0]++;
      NComponents[subDetIndex][component][1] += nFibers;
      NComponents[subDetIndex][component][2] += nAPVs;
      NComponents[subDetIndex][component][3] += nStrips;
    }
    delete reader;
  }

  // generic code to fill the vectors of bad components
  /*--------------------------------------------------------------------*/
  void fillBCArrays(const SiStripQuality* siStripQuality_,
                    int NTkBadComponent[4],
                    int NBadComponent[4][19][4],
                    const TrackerTopology m_trackerTopo)
  /*--------------------------------------------------------------------*/
  {
    std::vector<SiStripQuality::BadComponent> BC = siStripQuality_->getBadComponentList();

    for (size_t i = 0; i < BC.size(); ++i) {
      //&&&&&&&&&&&&&
      //Full Tk
      //&&&&&&&&&&&&&

      if (BC.at(i).BadModule)
        NTkBadComponent[0]++;
      if (BC.at(i).BadFibers)
        NTkBadComponent[1] +=
            ((BC.at(i).BadFibers >> 2) & 0x1) + ((BC.at(i).BadFibers >> 1) & 0x1) + ((BC.at(i).BadFibers) & 0x1);
      if (BC.at(i).BadApvs)
        NTkBadComponent[2] += ((BC.at(i).BadApvs >> 5) & 0x1) + ((BC.at(i).BadApvs >> 4) & 0x1) +
                              ((BC.at(i).BadApvs >> 3) & 0x1) + ((BC.at(i).BadApvs >> 2) & 0x1) +
                              ((BC.at(i).BadApvs >> 1) & 0x1) + ((BC.at(i).BadApvs) & 0x1);

      //&&&&&&&&&&&&&&&&&
      //Single SubSyste
      //&&&&&&&&&&&&&&&&&
      int component;
      DetId detectorId = DetId(BC.at(i).detid);
      int subDet = detectorId.subdetId();
      if (subDet == StripSubdetector::TIB) {
        //&&&&&&&&&&&&&&&&&
        //TIB
        //&&&&&&&&&&&&&&&&&

        component = m_trackerTopo.tibLayer(BC.at(i).detid);
        SiStripPI::setBadComponents(0, component, BC.at(i), NBadComponent);

      } else if (subDet == StripSubdetector::TID) {
        //&&&&&&&&&&&&&&&&&
        //TID
        //&&&&&&&&&&&&&&&&&

        component = m_trackerTopo.tidSide(BC.at(i).detid) == 2 ? m_trackerTopo.tidWheel(BC.at(i).detid)
                                                               : m_trackerTopo.tidWheel(BC.at(i).detid) + 3;
        SiStripPI::setBadComponents(1, component, BC.at(i), NBadComponent);

      } else if (subDet == StripSubdetector::TOB) {
        //&&&&&&&&&&&&&&&&&
        //TOB
        //&&&&&&&&&&&&&&&&&

        component = m_trackerTopo.tobLayer(BC.at(i).detid);
        SiStripPI::setBadComponents(2, component, BC.at(i), NBadComponent);

      } else if (subDet == StripSubdetector::TEC) {
        //&&&&&&&&&&&&&&&&&
        //TEC
        //&&&&&&&&&&&&&&&&&

        component = m_trackerTopo.tecSide(BC.at(i).detid) == 2 ? m_trackerTopo.tecWheel(BC.at(i).detid)
                                                               : m_trackerTopo.tecWheel(BC.at(i).detid) + 9;
        SiStripPI::setBadComponents(3, component, BC.at(i), NBadComponent);
      }
    }

    //&&&&&&&&&&&&&&&&&&
    // Single Strip Info
    //&&&&&&&&&&&&&&&&&&

    edm::FileInPath fp_ = edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat");
    SiStripDetInfoFileReader* reader = new SiStripDetInfoFileReader(fp_.fullPath());

    float percentage = 0;

    SiStripQuality::RegistryIterator rbegin = siStripQuality_->getRegistryVectorBegin();
    SiStripQuality::RegistryIterator rend = siStripQuality_->getRegistryVectorEnd();

    for (SiStripBadStrip::RegistryIterator rp = rbegin; rp != rend; ++rp) {
      uint32_t detid = rp->detid;

      int subdet = -999;
      int component = -999;
      DetId detectorId = DetId(detid);
      int subDet = detectorId.subdetId();
      if (subDet == StripSubdetector::TIB) {
        subdet = 0;
        component = m_trackerTopo.tibLayer(detid);
      } else if (subDet == StripSubdetector::TID) {
        subdet = 1;
        component =
            m_trackerTopo.tidSide(detid) == 2 ? m_trackerTopo.tidWheel(detid) : m_trackerTopo.tidWheel(detid) + 3;
      } else if (subDet == StripSubdetector::TOB) {
        subdet = 2;
        component = m_trackerTopo.tobLayer(detid);
      } else if (subDet == StripSubdetector::TEC) {
        subdet = 3;
        component =
            m_trackerTopo.tecSide(detid) == 2 ? m_trackerTopo.tecWheel(detid) : m_trackerTopo.tecWheel(detid) + 9;
      }

      SiStripQuality::Range sqrange = SiStripQuality::Range(siStripQuality_->getDataVectorBegin() + rp->ibegin,
                                                            siStripQuality_->getDataVectorBegin() + rp->iend);

      percentage = 0;
      for (int it = 0; it < sqrange.second - sqrange.first; it++) {
        unsigned int range = siStripQuality_->decode(*(sqrange.first + it)).range;
        NTkBadComponent[3] += range;
        NBadComponent[subdet][0][3] += range;
        NBadComponent[subdet][component][3] += range;
        percentage += range;
      }
      if (percentage != 0)
        percentage /= 128. * reader->getNumberOfApvsAndStripLength(detid).first;
      if (percentage > 1)
        edm::LogError("SiStripBadStrip_PayloadInspector")
            << "PROBLEM detid " << detid << " value " << percentage << std::endl;
    }

    delete reader;
  }

  /*--------------------------------------------------------------------*/
  void printBCDebug(int NTkBadComponent[4], int NBadComponent[4][19][4])
  /*--------------------------------------------------------------------*/
  {
    //&&&&&&&&&&&&&&&&&&
    // printout
    //&&&&&&&&&&&&&&&&&&

    std::stringstream ss;
    ss.str("");
    ss << "\n-----------------\nGlobal Info\n-----------------";
    ss << "\nBadComponent \t   Modules \tFibers "
          "\tApvs\tStrips\n----------------------------------------------------------------";
    ss << "\nTracker:\t\t" << NTkBadComponent[0] << "\t" << NTkBadComponent[1] << "\t" << NTkBadComponent[2] << "\t"
       << NTkBadComponent[3];
    ss << "\n";
    ss << "\nTIB:\t\t\t" << NBadComponent[0][0][0] << "\t" << NBadComponent[0][0][1] << "\t" << NBadComponent[0][0][2]
       << "\t" << NBadComponent[0][0][3];
    ss << "\nTID:\t\t\t" << NBadComponent[1][0][0] << "\t" << NBadComponent[1][0][1] << "\t" << NBadComponent[1][0][2]
       << "\t" << NBadComponent[1][0][3];
    ss << "\nTOB:\t\t\t" << NBadComponent[2][0][0] << "\t" << NBadComponent[2][0][1] << "\t" << NBadComponent[2][0][2]
       << "\t" << NBadComponent[2][0][3];
    ss << "\nTEC:\t\t\t" << NBadComponent[3][0][0] << "\t" << NBadComponent[3][0][1] << "\t" << NBadComponent[3][0][2]
       << "\t" << NBadComponent[3][0][3];
    ss << "\n";

    for (int i = 1; i < 5; ++i)
      ss << "\nTIB Layer " << i << " :\t\t" << NBadComponent[0][i][0] << "\t" << NBadComponent[0][i][1] << "\t"
         << NBadComponent[0][i][2] << "\t" << NBadComponent[0][i][3];
    ss << "\n";
    for (int i = 1; i < 4; ++i)
      ss << "\nTID+ Disk " << i << " :\t\t" << NBadComponent[1][i][0] << "\t" << NBadComponent[1][i][1] << "\t"
         << NBadComponent[1][i][2] << "\t" << NBadComponent[1][i][3];
    for (int i = 4; i < 7; ++i)
      ss << "\nTID- Disk " << i - 3 << " :\t\t" << NBadComponent[1][i][0] << "\t" << NBadComponent[1][i][1] << "\t"
         << NBadComponent[1][i][2] << "\t" << NBadComponent[1][i][3];
    ss << "\n";
    for (int i = 1; i < 7; ++i)
      ss << "\nTOB Layer " << i << " :\t\t" << NBadComponent[2][i][0] << "\t" << NBadComponent[2][i][1] << "\t"
         << NBadComponent[2][i][2] << "\t" << NBadComponent[2][i][3];
    ss << "\n";
    for (int i = 1; i < 10; ++i)
      ss << "\nTEC+ Disk " << i << " :\t\t" << NBadComponent[3][i][0] << "\t" << NBadComponent[3][i][1] << "\t"
         << NBadComponent[3][i][2] << "\t" << NBadComponent[3][i][3];
    for (int i = 10; i < 19; ++i)
      ss << "\nTEC- Disk " << i - 9 << " :\t\t" << NBadComponent[3][i][0] << "\t" << NBadComponent[3][i][1] << "\t"
         << NBadComponent[3][i][2] << "\t" << NBadComponent[3][i][3];
    ss << "\n";

    //edm::LogInfo("SiStripBadStrip_PayloadInspector") << ss.str() << std::endl;
    std::cout << ss.str() << std::endl;
  }

  enum palette { HALFGRAY, GRAY, BLUES, REDS, ANTIGRAY, FIRE, ANTIFIRE, LOGREDBLUE, BLUERED, LOGBLUERED, DEFAULT };

  /*--------------------------------------------------------------------*/
  void setPaletteStyle(SiStripPI::palette palette)
  /*--------------------------------------------------------------------*/
  {
    TStyle* palettestyle = new TStyle("palettestyle", "Style for P-TDR");

    const int NRGBs = 5;
    const int NCont = 255;

    switch (palette) {
      case HALFGRAY: {
        double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        double red[NRGBs] = {1.00, 0.91, 0.80, 0.67, 1.00};
        double green[NRGBs] = {1.00, 0.91, 0.80, 0.67, 1.00};
        double blue[NRGBs] = {1.00, 0.91, 0.80, 0.67, 1.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      } break;

      case GRAY: {
        double stops[NRGBs] = {0.00, 0.01, 0.05, 0.09, 0.1};
        double red[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
        double green[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
        double blue[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      } break;

      case BLUES: {
        double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        double red[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
        double green[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
        double blue[NRGBs] = {1.00, 1.00, 1.00, 1.00, 1.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);

      } break;

      case REDS: {
        double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        double red[NRGBs] = {1.00, 1.00, 1.00, 1.00, 1.00};
        double green[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
        double blue[NRGBs] = {1.00, 0.84, 0.61, 0.34, 0.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      } break;

      case ANTIGRAY: {
        double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        double red[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        double green[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        double blue[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      } break;

      case FIRE: {
        const int NCOLs = 4;
        double stops[NCOLs] = {0.00, 0.20, 0.80, 1.00};
        double red[NCOLs] = {1.00, 1.00, 1.00, 0.50};
        double green[NCOLs] = {1.00, 1.00, 0.00, 0.00};
        double blue[NCOLs] = {0.20, 0.00, 0.00, 0.00};
        TColor::CreateGradientColorTable(NCOLs, stops, red, green, blue, NCont);
      } break;

      case ANTIFIRE: {
        const int NCOLs = 4;
        double stops[NCOLs] = {0.00, 0.20, 0.80, 1.00};
        double red[NCOLs] = {0.50, 1.00, 1.00, 1.00};
        double green[NCOLs] = {0.00, 0.00, 1.00, 1.00};
        double blue[NCOLs] = {0.00, 0.00, 0.00, 0.20};
        TColor::CreateGradientColorTable(NCOLs, stops, red, green, blue, NCont);
      } break;

      case LOGREDBLUE: {
        double stops[NRGBs] = {0.0001, 0.0010, 0.0100, 0.1000, 1.0000};
        double red[NRGBs] = {1.00, 0.75, 0.50, 0.25, 0.00};
        double green[NRGBs] = {0.00, 0.00, 0.00, 0.00, 0.00};
        double blue[NRGBs] = {0.00, 0.25, 0.50, 0.75, 1.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      } break;

      case LOGBLUERED: {
        double stops[NRGBs] = {0.0001, 0.0010, 0.0100, 0.1000, 1.0000};
        double red[NRGBs] = {0.00, 0.25, 0.50, 0.75, 1.00};
        double green[NRGBs] = {0.00, 0.00, 0.00, 0.00, 0.00};
        double blue[NRGBs] = {1.00, 0.75, 0.50, 0.25, 0.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      } break;

      case BLUERED: {
        double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        double red[NRGBs] = {0.00, 0.25, 0.50, 0.75, 1.00};
        double green[NRGBs] = {0.00, 0.00, 0.00, 0.00, 0.00};
        double blue[NRGBs] = {1.00, 0.75, 0.50, 0.25, 0.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      } break;

      case DEFAULT: {
        double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
        double red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
        double green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
        double blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
        TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      } break;
      default:
        std::cout << "should nevere be here" << std::endl;
        break;
    }

    palettestyle->SetNumberContours(NCont);
  }

};  // namespace SiStripPI
#endif
