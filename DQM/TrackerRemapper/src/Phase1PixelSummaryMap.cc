#include "TArrow.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH2Poly.h"
#include "TLatex.h"
#include "TStyle.h"

#include <fmt/printf.h>
#include <fstream>
#include <memory>
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelSummaryMap.h"

// set option, but only if not already set
//============================================================================
void Phase1PixelSummaryMap::resetOption(const char* option) {
  if (m_option != nullptr && !m_option[0]) {
    m_option = option;
  } else {
    edm::LogError("Phase1PixelSummaryMap")
        << "Option has already been set to " << m_option << ". It's not possible to reset it.";
  }
}

//============================================================================
void Phase1PixelSummaryMap::createTrackerBaseMap() {
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
void Phase1PixelSummaryMap::printTrackerMap(TCanvas& canvas, const float topMargin, int index) {
  //canvas = TCanvas("c1","c1",plotWidth,plotHeight);
  if (index != 0)
    canvas.cd(index);
  else
    canvas.cd();

  if (index == 0) {
    canvas.SetTopMargin(topMargin);
    canvas.SetBottomMargin(0.02);
    canvas.SetLeftMargin(0.02);
    canvas.SetRightMargin(0.14);
  } else {
    m_BaseTrackerMap->GetZaxis()->SetTitleOffset(1.5);
    canvas.cd(index)->SetTopMargin(topMargin);
    canvas.cd(index)->SetBottomMargin(0.02);
    canvas.cd(index)->SetLeftMargin(0.02);
    canvas.cd(index)->SetRightMargin(0.14);
  }

  m_BaseTrackerMap->Draw("AL");
  m_BaseTrackerMap->Draw("AC COLZ0 L SAME");

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
  txt.SetTextSize(0.03);
  txt.DrawLatex(0.5, ((index == 0) ? 0.95 : 0.93), (fmt::sprintf("Pixel Tracker Map: %s", m_title)).c_str());
  txt.SetTextSize(0.03);

  txt.DrawLatex(0.55, 0.125, "-DISK");
  txt.DrawLatex(0.55, 0.875, "+DISK");

  txt.DrawLatex(0.08, 0.28, "+z");
  txt.DrawLatex(0.25, 0.70, "+phi");
  txt.DrawLatex((index == 0) ? 0.31 : 0.33, 0.78, "+x");
  txt.DrawLatex((index == 0) ? 0.21 : 0.22, ((index == 0) ? 0.96 : 0.94), "+y");

  txt.SetTextAngle(90);
  txt.DrawLatex(0.04, 0.5, "BARREL");

  edm::LogPrint("Phase1PixelSummaryMap") << "Base Tracker Map: printed" << std::endl;
}

//============================================================================
bool Phase1PixelSummaryMap::fillTrackerMap(unsigned int id, double value) {
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

//============================================================================
const std::pair<float, float> Phase1PixelSummaryMap::getZAxisRange() const {
  return std::make_pair(m_BaseTrackerMap->GetMinimum(), m_BaseTrackerMap->GetMaximum());
}

//============================================================================
void Phase1PixelSummaryMap::setZAxisRange(const double min, const double max) {
  m_BaseTrackerMap->GetZaxis()->SetRangeUser(min, max);
}

//============================================================================
void Phase1PixelSummaryMap::addNamedBins(
    edm::FileInPath geoFile, int tX, int tY, int sX, int sY, bool applyModuleRotation) {
  auto cornerFileName = geoFile.fullPath();
  std::ifstream cornerFile(cornerFileName.c_str());
  if (!cornerFile.good()) {
    throw cms::Exception("FileError") << "Problem opening corner file: " << cornerFileName;
  }
  std::string line;
  while (std::getline(cornerFile, line)) {
    if (!line.empty()) {
      std::istringstream iss(line);

      auto tokens = Ph1PMapSummaryHelper::tokenize(line, '"');
      // Printing the token vector
      for (unsigned int i = 0; i < tokens.size(); i++)
        LOGDEBUG("Phase1PixelSummaryMap") << tokens[i] << '\n';

      auto detInfo = Ph1PMapSummaryHelper::tokenize(tokens[0], ' ');
      unsigned int detId = stoi(detInfo[0]);
      std::string detIdName = detInfo[1];
      auto xy = Ph1PMapSummaryHelper::tokenize(tokens[1], ' ');
      unsigned int verNum = 1;
      std::vector<float> xP, yP;
      for (const auto& coord : xy) {
        auto coordSpl = Ph1PMapSummaryHelper::tokenize(coord, ',');
        if (applyModuleRotation) {
          xP.push_back(-(std::stof(coordSpl[0]) * sX + tX));
          yP.push_back(((std::stof(coordSpl[1]) * sY + tY)));
        } else {
          xP.push_back(std::stof(coordSpl[0]) * sX + tX);
          yP.push_back(std::stof(coordSpl[1]) * sY + tY);
        }
        verNum++;
      }
      //close the polygon
      xP.push_back(xP[0]);
      yP.push_back(yP[0]);

      LOGDEBUG("Phase1PixelSummaryMap") << detId << "[";
      for (const auto& p : xP) {
        LOGDEBUG("Phase1PixelSummaryMap") << p << ",";
      }
      LOGDEBUG("Phase1PixelSummaryMap") << "] [ ";
      for (const auto& q : yP) {
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
