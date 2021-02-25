#ifndef DQM_TRACKERREMAPPER_PHASE1PIXELSUMMARYMAP_H
#define DQM_TRACKERREMAPPER_PHASE1PIXELSUMMARYMAP_H

#include "TArrow.h"
#include "TH2Poly.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

#include <fmt/printf.h>
#include <fstream>
#include <memory>
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"

#ifndef PH1PSUMMARYMAP_STANDALONE
#define LOGDEBUG(x) LogDebug(x)
#define LOGINFO(x) edm::LogInfo(x)
#define LOGPRINT(x) edm::LogPrint(x)
#else
#define LOGDEBUG(x) std::cout << x << " Debug : "
#define LOGINFO(x) std::cout << x << " Info : "
#define LOGPRINT(x) std::cout << x << " : "
#endif

using indexedCorners = std::map<unsigned int, std::pair<std::vector<float>, std::vector<float>>>;

/*--------------------------------------------------------------------
/ Ancillary class to build pixel phase-1 tracker maps
/--------------------------------------------------------------------*/
class Phase1PixelSummaryMap {
public:
  Phase1PixelSummaryMap(const char* option, std::string title, std::string zAxisTitle)
      : m_option{option},
        m_title{title},
        m_zAxisTitle{zAxisTitle},
        m_trackerTopo{StandaloneTrackerTopology::fromTrackerParametersXMLFile(
            edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml").fullPath())} {
    // store the file in path for the corners (BPIX)
    for (unsigned int i = 1; i <= 4; i++) {
      m_cornersBPIX.push_back(edm::FileInPath(Form("DQM/SiStripMonitorClient/data/Geometry/vertices_barrel_%i", i)));
    }

    // store the file in path for the corners (BPIX)
    for (int j : {-3, -2, -1, 1, 2, 3}) {
      m_cornersFPIX.push_back(edm::FileInPath(Form("DQM/SiStripMonitorClient/data/Geometry/vertices_forward_%i", j)));
    }
  }

  ~Phase1PixelSummaryMap() {}

  // set option, but only if not already set
  void resetOption(const char* option) {
    if (m_option != nullptr && !m_option[0]) {
      m_option = option;
    } else {
      edm::LogError("Phase1PixelSummaryMap")
          << "Option has already been set to " << m_option << ". It's not possible to reset it.";
    }
  }

  //============================================================================
  void createTrackerBaseMap() {
    m_BaseTrackerMap = std::make_shared<TH2Poly>("Summary", "", -10, 160, -70, 70);
    m_BaseTrackerMap->SetFloat(true);
    m_BaseTrackerMap->GetXaxis()->SetTitle("");
    m_BaseTrackerMap->GetYaxis()->SetTitle("");
    m_BaseTrackerMap->GetZaxis()->SetTitle(m_zAxisTitle.c_str());
    m_BaseTrackerMap->GetZaxis()->CenterTitle();
    m_BaseTrackerMap->GetZaxis()->SetTitleOffset(1.2);
    m_BaseTrackerMap->SetOption("COLZ L");
    m_BaseTrackerMap->SetStats(false);

    //BARREL FIRST
    for (unsigned int i = 0; i < maxPxBarrel; i++) {
      LOGINFO("Phase1PixelSummaryMap") << "barrel, shift: " << i << " corner: " << i << std::endl;
      LOGINFO("Phase1PixelSummaryMap") << "translate x: " << 0 << std::endl;
      LOGINFO("Phase1PixelSummaryMap") << "translate y: " << barrelLadderShift[i] << std::endl;

      int currBarrelTranslateX = 0;
      int currBarrelTranslateY = barrelLadderShift[i];
      addNamedBins(m_cornersBPIX[i], currBarrelTranslateX, currBarrelTranslateY, 1, 1, true);
    }

    //MINUS FORWARD
    for (int j : {-3, -2, -1}) {
      LOGINFO("Phase1PixelSummaryMap") << "negative fwd, shift: " << -j - 1 << " corner: " << maxPxForward + j
                                       << std::endl;
      LOGINFO("Phase1PixelSummaryMap") << "translate x: " << forwardDiskXShift[-j - 1] << std::endl;
      LOGINFO("Phase1PixelSummaryMap") << "translate y: " << -forwardDiskYShift << std::endl;

      int currForwardTranslateX = forwardDiskXShift[-j - 1];
      int currForwardTranslateY = -forwardDiskYShift;
      addNamedBins(m_cornersFPIX[maxPxForward + j], currForwardTranslateX, currForwardTranslateY, 1, 1);
    }

    //PLUS FORWARD
    for (int k : {1, 2, 3}) {
      LOGINFO("Phase1PixelSummaryMap") << "positive fwd, shift: " << k << " corner: " << maxPxForward + k - 1
                                       << std::endl;
      LOGINFO("Phase1PixelSummaryMap") << "translate x: " << forwardDiskXShift[k - 1] << std::endl;
      LOGINFO("Phase1PixelSummaryMap") << "translate y: " << forwardDiskYShift << std::endl;

      int currForwardTranslateX = forwardDiskXShift[k - 1];
      int currForwardTranslateY = forwardDiskYShift;
      addNamedBins(m_cornersFPIX[maxPxForward + k - 1], currForwardTranslateX, currForwardTranslateY, 1, 1);
    }

    edm::LogPrint("Phase1PixelSummaryMap") << "Base Tracker Map: constructed" << std::endl;
    return;
  }

  //============================================================================
  void printTrackerMap(TCanvas& canvas) {
    //canvas = TCanvas("c1","c1",plotWidth,plotHeight);
    canvas.cd();
    canvas.SetTopMargin(0.02);
    canvas.SetBottomMargin(0.02);
    canvas.SetLeftMargin(0.02);
    canvas.SetRightMargin(0.14);
    m_BaseTrackerMap->Draw("AC COLZ L");

    //### z arrow
    arrow = TArrow(0.05, 27.0, 0.05, -30.0, 0.02, "|>");
    arrow.SetLineWidth(4);
    arrow.Draw();
    arrow.SetAngle(30);
    //### phi arrow
    phiArrow = TArrow(0.0, 27.0, 30.0, 27.0, 0.02, "|>");
    phiArrow.SetLineWidth(4);
    phiArrow.Draw();
    phiArrow.SetAngle(30);
    //### x arrow
    xArrow = TArrow(25.0, 44.5, 50.0, 44.5, 0.02, "|>");
    xArrow.SetLineWidth(4);
    xArrow.Draw();
    xArrow.SetAngle(30);
    //### y arrow
    yArrow = TArrow(25.0, 44.5, 25.0, 69.5, 0.02, "|>");
    yArrow.SetLineWidth(4);
    yArrow.Draw();
    yArrow.SetAngle(30);

    //###################################################
    //# add some captions
    auto txt = TLatex();
    txt.SetNDC();
    txt.SetTextFont(1);
    txt.SetTextColor(1);
    txt.SetTextAlign(22);
    txt.SetTextAngle(0);

    //# draw new-style title
    txt.SetTextSize(0.05);
    txt.DrawLatex(0.5, 0.95, (fmt::sprintf("Pixel Tracker Map: %s", m_title)).c_str());
    txt.SetTextSize(0.03);

    txt.DrawLatex(0.55, 0.125, "-DISK");
    txt.DrawLatex(0.55, 0.875, "+DISK");

    txt.DrawLatex(0.08, 0.28, "+z");
    txt.DrawLatex(0.25, 0.70, "+phi");
    txt.DrawLatex(0.31, 0.78, "+x");
    txt.DrawLatex(0.21, 0.96, "+y");

    txt.SetTextAngle(90);
    txt.DrawLatex(0.04, 0.5, "BARREL");

    edm::LogPrint("Phase1PixelSummaryMap") << "Base Tracker Map: printed" << std::endl;
  }

  //============================================================================
  template <typename type>
  bool fillTrackerMap(unsigned int id, type value) {
    auto detid = DetId(id);
    if (detid.subdetId() != PixelSubdetector::PixelBarrel && detid.subdetId() != PixelSubdetector::PixelEndcap) {
      edm::LogError("Phase1PixelSummaryMap")
          << __func__ << " The following detid " << id << " is not Pixel!" << std::endl;
      return false;
    } else {
      m_BaseTrackerMap->Fill(TString::Format("%u", id), value);
      return true;
    }
  }

protected:
  //============================================================================
  // utility to tokenize std::string
  //============================================================================
  std::vector<std::string> myTokenize(std::string line, char delimiter) {
    // Vector of string to save tokens
    std::vector<std::string> tokens;
    std::stringstream check1(line);
    std::string intermediate;

    // Tokenizing w.r.t. delimiter
    while (getline(check1, intermediate, delimiter)) {
      tokens.push_back(intermediate);
    }
    return tokens;
  }

  //============================================================================
  void addNamedBins(edm::FileInPath geoFile, int tX, int tY, int sX, int sY, bool applyModuleRotation = false) {
    auto cornerFileName = geoFile.fullPath();
    std::ifstream cornerFile(cornerFileName.c_str());
    if (!cornerFile.good()) {
      throw cms::Exception("FileError") << "Problem opening corner file: " << cornerFileName;
    }
    std::string line;
    while (std::getline(cornerFile, line)) {
      if (!line.empty()) {
        std::istringstream iss(line);

        auto tokens = this->myTokenize(line, '"');
        // Printing the token vector
        for (unsigned int i = 0; i < tokens.size(); i++)
          LOGDEBUG("Phase1PixelSummaryMap") << tokens[i] << '\n';

        auto detInfo = this->myTokenize(tokens[0], ' ');
        unsigned int detId = stoi(detInfo[0]);
        std::string detIdName = detInfo[1];
        auto xy = this->myTokenize(tokens[1], ' ');
        unsigned int verNum = 1;
        std::vector<float> xP, yP;
        for (const auto& coord : xy) {
          auto coordSpl = myTokenize(coord, ',');
          if (applyModuleRotation) {
            xP.push_back(-(stof(coordSpl[0]) * sX + tX));
            yP.push_back(((stof(coordSpl[1]) * sY + tY)));
          } else {
            xP.push_back(stof(coordSpl[0]) * sX + tX);
            yP.push_back(stof(coordSpl[1]) * sY + tY);
          }
          verNum++;
        }
        //close the polygon
        xP.push_back(xP[0]);
        yP.push_back(yP[0]);

        LOGDEBUG("Phase1PixelSummaryMap") << detId << "[";
        for (auto p : xP) {
          LOGDEBUG("Phase1PixelSummaryMap") << p << ",";
        }
        LOGDEBUG("Phase1PixelSummaryMap") << "] [ ";
        for (auto q : yP) {
          LOGDEBUG("Phase1PixelSummaryMap") << q << ",";
        }
        LOGDEBUG("Phase1PixelSummaryMap") << "]" << std::endl;

        const unsigned int N = verNum;
        if (applyModuleRotation) {
          bins[detId] = std::make_shared<TGraph>(N, &yP[0], &xP[0]);
        } else {
          bins[detId] = std::make_shared<TGraph>(N, &xP[0], &yP[0]);
          //bins[detId] = std::make_shared<TGraph>(N, &yP[0], &xP[0]); // rotation by 90 deg (so that it had the same layout as for the strips)
        }

        bins[detId]->SetName(detInfo[0].c_str());
        m_BaseTrackerMap->AddBin(bins[detId]->Clone());
      }
    }
    return;
  }

private:
  Option_t* m_option;
  const std::string m_title;
  const std::string m_zAxisTitle;

  TrackerTopology m_trackerTopo;
  std::shared_ptr<TH2Poly> m_BaseTrackerMap;
  std::map<uint32_t, std::shared_ptr<TGraph>> bins;

  std::vector<edm::FileInPath> m_cornersBPIX;
  std::vector<edm::FileInPath> m_cornersFPIX;

  static const unsigned int maxPxBarrel = 4;
  static const unsigned int maxPxForward = 3;
  const std::array<int, maxPxBarrel> barrelLadderShift = {{0, 14, 44, 90}};
  const std::array<int, maxPxForward> forwardDiskXShift = {{25, 75, 125}};

  const int forwardDiskYShift = 45;  //# to make +DISK on top in the 'strip-like' layout

  const int plotWidth = 3000;
  const int plotHeight = 2000;

  TArrow arrow, phiArrow, xArrow, yArrow;
};

#endif
