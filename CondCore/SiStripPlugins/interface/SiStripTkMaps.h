#ifndef CONDCORE_SISTRIPPLUGINS_SISTRIPTKMAPS_H
#define CONDCORE_SISTRIPPLUGINS_SISTRIPTKMAPS_H

// CMSSW includes
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// ROOT includes
#include "TPaletteAxis.h"
#include "TGaxis.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TLatex.h"
#include "TH2Poly.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TProfile2D.h"
#include "TStyle.h"

// STL includes
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>

#define MYOUT LogDebug("SiStripTkMaps")

/*--------------------------------------------------------------------
/ Ancillary class to build SiStrip Tracker maps
/--------------------------------------------------------------------*/
class SiStripTkMaps {
public:
  SiStripTkMaps(const char* option)
      : m_option{option},
        m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
            edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {}

  ~SiStripTkMaps() {}

  //============================================================================
  void bookMap(const std::string mapTitle,const std::string zAxisTitle) {
    double minx = 0xFFFFFF, maxx = -0xFFFFFF, miny = 0xFFFFFF, maxy = -0xFFFFFFF;
    readVertices(minx, maxx, miny, maxy);

    // set the titles
    m_zAxisTitle=zAxisTitle;
    m_mapTitle=mapTitle;

    TGaxis::SetMaxDigits(2);

    static constexpr int margin = 5;
    m_trackerMap =
        new TH2Poly("Tracker Map", m_mapTitle.c_str(), minx - margin, maxx + margin, miny - margin, maxy + margin);
    m_trackerMap->SetFloat();
    m_trackerMap->SetOption(m_option);
    m_trackerMap->SetStats(false);
    m_trackerMap->GetZaxis()->SetLabelSize(0.03);
    m_trackerMap->GetZaxis()->SetTitleOffset(0.7);
    m_trackerMap->GetZaxis()->SetTitle(m_zAxisTitle.c_str());
    m_trackerMap->GetZaxis()->CenterTitle();

    for (const auto& pair : m_bins) {
      m_trackerMap->AddBin(pair.second->Clone());
    }
  }

  //============================================================================
  void fill(long rawid, double val) { m_trackerMap->Fill(TString::Format("%ld", rawid), val); }

  //============================================================================
  void drawMap(TCanvas& canvas, std::string option = "") {
    canvas.cd();
    adjustCanvasMargins(canvas.cd(), 0.07, 0.01, 0.01, 0.10);

    m_trackerMap->SetTitle("");  
    if (!option.empty()) {
      m_trackerMap->Draw(option.c_str());
    } else {
      m_trackerMap->Draw();
    }

    canvas.SetFrameLineColor(0);

  }

  //============================================================================
  TH2Poly* getTheMap() { return m_trackerMap; }

  //============================================================================
  void setZAxisRange(double xmin, double xmax) { m_trackerMap->GetZaxis()->SetRangeUser(xmin, xmax); }

  //============================================================================
  void adjustCanvasMargins(TVirtualPad* pad, float top, float bottom, float left, float right) {
    if (top > 0) {
      pad->SetTopMargin(top);
    }
    if (bottom > 0) {
      pad->SetBottomMargin(bottom);
    }
    if (left > 0) {
      pad->SetLeftMargin(left);
    }
    if (right > 0) {
      pad->SetRightMargin(right);
    }
  }

  //============================================================================
  void readVertices(double& minx, double& maxx, double& miny, double& maxy) {
    std::ifstream in;

    in.open(edm::FileInPath("DQM/SiStripMonitorClient/data/Geometry/tracker_map_bare").fullPath().c_str());

    if (!in.good()) {
      throw cms::Exception("FileError") << "Problem opening corner file!!" << std::endl;
      return;
    }

    while (in.good()) {
      long detid = 0;
      double x[5], y[5];

      std::string line;
      std::getline(in, line);

      TString string(line);
      TObjArray* array = string.Tokenize(" ");
      int ix{0}, iy{0};
      bool isPixel{false};
      for (int i = 0; i < array->GetEntries(); ++i) {
        if (i == 0) {
          detid = static_cast<TObjString*>(array->At(i))->String().Atoll();

          // Drop Pixel Data
          DetId detId(detid);
          if (detId.subdetId() == PixelSubdetector::PixelBarrel || detId.subdetId() == PixelSubdetector::PixelEndcap) {
            isPixel = true;
            break;
          }
        } else {
          if (i % 2 == 0) {
            x[ix] = static_cast<TObjString*>(array->At(i))->String().Atof();

            if (x[ix] < minx) {
              minx = x[ix];
            }

            if (x[ix] > maxx) {
              maxx = x[ix];
            }

            ++ix;
          } else {
            y[iy] = static_cast<TObjString*>(array->At(i))->String().Atof();

            if (y[iy] < miny) {
              miny = y[iy];
            }
            if (y[iy] > maxy) {
              maxy = y[iy];
            }

            ++iy;
          }
        }
      }

      if (isPixel) {
        continue;
      }

      m_detIdVector.push_back(detid);
      m_bins[detid] = std::make_shared<TGraph>(ix, x, y);
      m_bins[detid]->SetName(TString::Format("%ld", detid));
      m_bins[detid]->SetTitle(TString::Format("Module ID=%ld", detid));
    }
  }

private:
  Option_t* m_option;
  std::string m_mapTitle="";
  std::string m_zAxisTitle="";
  std::map<long, std::shared_ptr<TGraph> > m_bins;
  std::vector<unsigned> m_detIdVector;
  TrackerTopology m_trackerTopo;
  TH2Poly* m_trackerMap{nullptr};
};

#endif
