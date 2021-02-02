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
#include "TArrow.h"
#include "TPaletteAxis.h"
#include "TGaxis.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TGraph.h"
#include "TLatex.h"
#include "TH2Poly.h"
#include "TStyle.h"

// STL includes
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// boost includes
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

  ~SiStripTkMaps() = default;

  //============================================================================
  void bookMap(const std::string mapTitle, const std::string zAxisTitle) {
    double minx = 0xFFFFFF, maxx = -0xFFFFFF, miny = 0xFFFFFF, maxy = -0xFFFFFFF;
    readVertices(minx, maxx, miny, maxy);

    // set the titles
    m_zAxisTitle = zAxisTitle;
    m_mapTitle = mapTitle;

    TGaxis::SetMaxDigits(2);

    // margin of the box
    static constexpr int margin = 5;
    m_trackerMap =
        new TH2Poly("Tracker Map", m_mapTitle.c_str(), minx - margin, maxx + margin, miny - margin, maxy + margin);
    m_trackerMap->SetFloat();
    m_trackerMap->SetOption(m_option);
    m_trackerMap->SetStats(false);
    m_trackerMap->GetZaxis()->SetLabelSize(0.03);
    m_trackerMap->GetZaxis()->SetTitleOffset(0.5);
    m_trackerMap->GetZaxis()->SetTitleSize(0.05);
    m_trackerMap->GetZaxis()->SetTitle(m_zAxisTitle.c_str());
    m_trackerMap->GetZaxis()->CenterTitle();

    for (const auto& pair : m_bins) {
      m_trackerMap->AddBin(pair.second->Clone());
    }
  }

  //============================================================================
  void fill(long rawid, double val) {
    m_trackerMap->Fill(TString::Format("%ld", rawid), val);
    m_values.push_back(val);
  }

  //============================================================================
  void drawMap(TCanvas& canvas, std::string option = "") {
    // margins
    static constexpr float tmargin_ = 0.08;
    static constexpr float bmargin_ = 0.02;
    static constexpr float lmargin_ = 0.02;
    static constexpr float rmargin_ = 0.08;

    // window size
    static constexpr int wH_ = 3000;
    static constexpr int hH_ = 850;

    canvas.cd();
    adjustCanvasMargins(canvas.cd(), tmargin_, bmargin_, lmargin_, rmargin_);
    canvas.Update();

    m_trackerMap->SetTitle("");
    if (!option.empty()) {
      m_trackerMap->Draw(option.c_str());
    } else {
      m_trackerMap->Draw();
    }

    canvas.SetFrameLineColor(0);
    gPad->Update();
    TPaletteAxis* palette = (TPaletteAxis*)m_trackerMap->GetListOfFunctions()->FindObject("palette");
    if (palette != nullptr) {
      palette->SetLabelSize(0.02);
      palette->SetX1NDC(1 - rmargin_);
      palette->SetX2NDC(1 - rmargin_ + lmargin_);
    }

    // if not right size, and not drawn in same mode
    if (canvas.GetWindowHeight() != hH_ && canvas.GetWindowWidth() != wH_ && option.find("same") == std::string::npos) {
      canvas.SetWindowSize(wH_, hH_);
    }

    // call the map dressing
    dressMap(canvas);
  }

  //============================================================================
  const TH2Poly* getTheMap() { return m_trackerMap; }

  //============================================================================
  inline const std::string& getTheMapTitle() { return m_mapTitle; }

  //============================================================================
  inline const std::string& getTheZAxisTitle() { return m_zAxisTitle; }

  //============================================================================
  inline const std::vector<unsigned int>& getTheFilledIds() { return m_detIdVector; }

  //============================================================================
  inline const std::vector<double>& getTheFilledValues() { return m_values; }

  //============================================================================
  void setZAxisRange(double xmin, double xmax) { m_trackerMap->GetZaxis()->SetRangeUser(xmin, xmax); }

private:
  // private members
  Option_t* m_option;
  std::string m_mapTitle = "";
  std::string m_zAxisTitle = "";
  double m_axmin, m_axmax;
  std::map<long, std::shared_ptr<TGraph>> m_bins;
  std::vector<unsigned int> m_detIdVector;
  std::vector<double> m_values;
  TrackerTopology m_trackerTopo;
  TH2Poly* m_trackerMap{nullptr};

  // private methods
  //============================================================================
  void dressMap(TCanvas& canv) {
    std::array<std::string, 12> barrelNames = {
        {"TIB L2", "TIB L1", "TIB L4", "TIB L3", "TOB L2", "TOB L1", "TOB L4", " TOB L3", "TOB L6", "TOB L5"}};
    std::array<std::string, 4> endcapNames = {{"TID", "TEC", "TID", "TEC"}};
    std::array<std::string, 24> disknumbering = {{"+1", "+2", "+3", "+1", "+2", "+3", "+4", "+5",
                                                  "+6", "+7", "+8", "+9", "-1", "-2", "-3", "-1",
                                                  "-2", "-3", "-4", "-5", "-6", "-7", "-8", "-9"}};

    static constexpr std::array<float, 12> b_coordx = {
        {0.1, 0.1, 0.26, 0.26, 0.41, 0.41, 0.56, 0.56, 0.725, 0.725, 0.05, 0.17}};
    static constexpr std::array<float, 12> b_coordy = {
        {0.70, 0.45, 0.70, 0.45, 0.70, 0.46, 0.70, 0.46, 0.70, 0.46, 0.85, 0.85}};

    static constexpr std::array<float, 4> e_coordx = {{0.01, 0.21, 0.01, 0.21}};
    static constexpr std::array<float, 4> e_coordy = {{0.89, 0.89, 0.17, 0.17}};

    static constexpr std::array<float, 24> n_coordx = {{0.01,  0.087, 0.165, 0.227, 0.305, 0.383, 0.461, 0.539,
                                                        0.616, 0.694, 0.772, 0.850, 0.01,  0.087, 0.165, 0.227,
                                                        0.305, 0.383, 0.461, 0.539, 0.617, 0.695, 0.773, 0.851}};

    static constexpr std::array<float, 24> n_coordy = {{0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85,
                                                        0.85, 0.85, 0.85, 0.85, 0.13, 0.13, 0.13, 0.13,
                                                        0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13}};

    canv.cd();
    for (const auto& name : barrelNames | boost::adaptors::indexed(0)) {
      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.035);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(b_coordx[name.index()], b_coordy[name.index()], name.value().c_str());
    }

    for (const auto& name : endcapNames | boost::adaptors::indexed(0)) {
      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.05);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(e_coordx[name.index()], e_coordy[name.index()], name.value().c_str());
    }

    for (const auto& name : disknumbering | boost::adaptors::indexed(0)) {
      auto ltx = TLatex();
      ltx.SetTextFont(62);
      ltx.SetTextSize(0.035);
      ltx.SetTextAlign(11);
      ltx.DrawLatexNDC(n_coordx[name.index()], n_coordy[name.index()], name.value().c_str());
    }

    auto ltx = TLatex();
    ltx.SetTextFont(62);
    ltx.SetTextSize(0.045);
    ltx.SetTextAlign(11);
    ltx.DrawLatexNDC(gPad->GetLeftMargin(), 1 - gPad->GetTopMargin() + 0.03, m_mapTitle.c_str());

    // barrel axes
    drawArrows(0.09, 0.23, 0.24, 0.45, "#phi", "z");
    // endcap axes
    drawArrows(0.85, 0.89, 0.83, 0.95, "x", "y");

    canv.Modified();
    canv.Update();  // make sure it's really (re)drawn
  }

  //============================================================================
  void drawArrows(const float x_X1,
                  const float x_X2,
                  const float x_Y1,
                  const float y_Y2,
                  const char* x_label,
                  const char* y_label) {
    auto arrow_X = TArrow();
    arrow_X.SetLineColor(kBlue);
    arrow_X.SetLineWidth(2);
    arrow_X.SetOption("|>");
    arrow_X.SetArrowSize(10);
    arrow_X.DrawLineNDC(x_X1, x_Y1, x_X2, x_Y1);

    auto arrow_Y = TArrow();
    arrow_Y.SetLineColor(kBlue);
    arrow_Y.SetLineWidth(2);
    arrow_Y.SetOption("|>");
    arrow_Y.SetArrowSize(10);
    arrow_Y.DrawLineNDC(x_X2, x_Y1, x_X2, y_Y2);

    auto text_X = TLatex();
    text_X.SetTextSize(0.04);
    text_X.SetTextAlign(11);
    text_X.SetTextColor(kBlue);
    text_X.DrawLatexNDC(x_X1, x_Y1 - 0.03, x_label);

    auto text_Y = TLatex();
    text_Y.SetTextSize(0.04);
    text_Y.SetTextAlign(11);
    text_Y.SetTextColor(kBlue);
    text_Y.DrawLatexNDC(x_X2 + 0.005, y_Y2 - 0.01, y_label);
  }

  //============================================================================
  void adjustCanvasMargins(TVirtualPad* pad, const float top, const float bottom, const float left, const float right) {
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

    // TPolyline vertices stored at https://github.com/cms-data/DQM-SiStripMonitorClient
    in.open(edm::FileInPath("DQM/SiStripMonitorClient/data/Geometry/tracker_map_bare").fullPath().c_str());

    if (!in.good()) {
      throw cms::Exception("FileError") << "SiStripTkMaps: problem opening vertices file!!" << std::endl;
      return;
    }

    while (in.good()) {
      long detid = 0;
      double x[5], y[5];

      std::string line;
      std::getline(in, line);
      typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
      boost::char_separator<char> sep{" "};
      tokenizer tok{line, sep};

      int ix{0}, iy{0};
      bool isPixel{false};
      for (const auto& t : tok | boost::adaptors::indexed(0)) {
        int i = t.index();
        if (i == 0) {
          detid = atoll((t.value()).c_str());

          // Drop Pixel Data
          DetId detId(detid);
          if (detId.subdetId() == PixelSubdetector::PixelBarrel || detId.subdetId() == PixelSubdetector::PixelEndcap) {
            isPixel = true;
            break;
          }
        } else {
          if (i % 2 == 0) {
            x[ix] = atof((t.value()).c_str());
            if (x[ix] < minx) {
              minx = x[ix];
            }
            if (x[ix] > maxx) {
              maxx = x[ix];
            }
            ++ix;
          } else {
            y[iy] = atof((t.value()).c_str());
            if (y[iy] < miny) {
              miny = y[iy];
            }
            if (y[iy] > maxy) {
              maxy = y[iy];
            }
            ++iy;
          }  // else
        }    // else
      }      // loop on entries

      if (isPixel) {
        continue;
      }

      m_bins[detid] = std::make_shared<TGraph>(ix, x, y);
      m_bins[detid]->SetName(TString::Format("%ld", detid));
      m_bins[detid]->SetTitle(TString::Format("Module ID=%ld", detid));
      m_detIdVector.push_back(detid);
    }
  }
};

#endif
