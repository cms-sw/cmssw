#include "TH2Poly.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"

#include <fmt/printf.h>
#include <fstream>
#include <iostream>
#include <boost/tokenizer.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "DQM/TrackerRemapper/interface/Phase1PixelMaps.h"

// set option, but only if not already set
//============================================================================
void Phase1PixelMaps::resetOption(const char* option) {
  if (m_option != nullptr && !m_option[0]) {
    m_option = option;
  } else {
    edm::LogError("Phase1PixelMaps") << "Option has already been set to " << m_option
                                     << ". It's not possible to reset it.";
  }
}

//============================================================================
void Phase1PixelMaps::bookBarrelHistograms(const std::string& currentHistoName, const char* what, const char* zaxis) {
  std::string histName;
  std::shared_ptr<TH2Poly> th2p;

  // check if the passed histogram name already exists, if not store it
  if (std::find(m_knownNames.begin(), m_knownNames.end(), currentHistoName) == m_knownNames.end()) {
    m_knownNames.emplace_back(currentHistoName);
  }

  for (unsigned i = 0; i < 4; ++i) {
    histName = "barrel_layer_";

    th2p = std::make_shared<TH2Poly>(
        (histName + std::to_string(i + 1)).c_str(), Form("PXBMap of %s - Layer %i", what, i + 1), -15.0, 15.0, 0.0, 5.0);

    th2p->SetFloat();

    th2p->GetXaxis()->SetTitle("z [cm]");
    th2p->GetYaxis()->SetTitle("ladder");
    th2p->GetZaxis()->SetTitle(zaxis);
    th2p->GetZaxis()->CenterTitle();
    th2p->SetStats(false);
    th2p->SetOption(m_option);
    pxbTh2PolyBarrel[currentHistoName].push_back(th2p);
  }

  th2p = std::make_shared<TH2Poly>("barrel_summary", Form("Barrel Pixel Map of %s", what), -5.0, 5.0, 0.0, 15.0);
  th2p->SetFloat();

  th2p->GetXaxis()->SetTitle("");
  th2p->GetYaxis()->SetTitle("");
  th2p->GetZaxis()->SetTitle(zaxis);
  th2p->GetZaxis()->CenterTitle();
  th2p->SetStats(false);
  th2p->SetOption(m_option);
  pxbTh2PolyBarrelSummary[currentHistoName] = th2p;

  // book the bins
  bookBarrelBins(currentHistoName);

  // set the isBooked bit to true;
  m_isBooked.first = true;
}

//============================================================================
void Phase1PixelMaps::bookForwardHistograms(const std::string& currentHistoName, const char* what, const char* zaxis) {
  std::string histName;
  std::shared_ptr<TH2Poly> th2p;

  // check if the passed histogram name already exists, if not store it
  if (std::find(m_knownNames.begin(), m_knownNames.end(), currentHistoName) == m_knownNames.end()) {
    m_knownNames.emplace_back(currentHistoName);
  }

  for (unsigned side = 1; side <= 2; ++side) {
    for (unsigned disk = 1; disk <= 3; ++disk) {
      histName = "forward_disk_";

      th2p = std::make_shared<TH2Poly>((histName + std::to_string((side == 1 ? -(int(disk)) : (int)disk))).c_str(),
                                       Form("PXFMap of %s - Side %i Disk %i", what, side, disk),
                                       -15.0,
                                       15.0,
                                       -15.0,
                                       15.0);
      th2p->SetFloat();
      th2p->GetXaxis()->SetTitle("x [cm]");
      th2p->GetYaxis()->SetTitle("y [cm]");
      th2p->GetZaxis()->SetTitle(zaxis);
      th2p->GetZaxis()->CenterTitle();
      th2p->SetStats(false);
      th2p->SetOption(m_option);
      pxfTh2PolyForward[currentHistoName].push_back(th2p);
    }
  }

  th2p = std::make_shared<TH2Poly>("forward_summary", Form("Forward Pixel Map of %s", what), -40.0, 40.0, -20.0, 90.0);
  th2p->SetFloat();

  th2p->GetXaxis()->SetTitle("");
  th2p->GetYaxis()->SetTitle("");
  th2p->GetZaxis()->SetTitle(zaxis);
  th2p->GetZaxis()->CenterTitle();
  th2p->SetStats(false);
  th2p->SetOption(m_option);
  pxfTh2PolyForwardSummary[currentHistoName] = th2p;

  // book the bins
  bookForwardBins(currentHistoName);

  m_isBooked.second = true;
}

//============================================================================
void Phase1PixelMaps::bookBarrelBins(const std::string& currentHistoName) {
  auto theIndexedCorners = this->retrieveCorners(m_cornersBPIX, 4);

  for (const auto& entry : theIndexedCorners) {
    auto id = entry.first;
    auto detid = DetId(id);
    if (detid.subdetId() != PixelSubdetector::PixelBarrel)
      continue;

    int layer = m_trackerTopo.pxbLayer(detid);
    int ladder = m_trackerTopo.pxbLadder(detid);

    auto theVectX = entry.second.first;
    auto theVectY = entry.second.second;

    float vertX[] = {theVectX[0], theVectX[1], theVectX[2], theVectX[3], theVectX[4]};
    float vertY[] = {(ladder - 1.0f), (ladder - 1.0f), (float)ladder, (float)ladder, (ladder - 1.0f)};

    bins[id] = std::make_shared<TGraph>(5, vertX, vertY);
    bins[id]->SetName(TString::Format("%u", id));

    // Summary plot
    for (unsigned k = 0; k < 5; ++k) {
      vertX[k] += ((layer == 2 || layer == 3) ? 0.0f : -60.0f);
      vertY[k] += ((layer > 2) ? 30.0f : 0.0f);
    }

    binsSummary[id] = std::make_shared<TGraph>(5, vertX, vertY);
    binsSummary[id]->SetName(TString::Format("%u", id));

    if (pxbTh2PolyBarrel.find(currentHistoName) != pxbTh2PolyBarrel.end()) {
      pxbTh2PolyBarrel[currentHistoName][layer - 1]->AddBin(bins[id]->Clone());
    } else {
      throw cms::Exception("LogicError") << currentHistoName << " is not found in the Barrel map! Aborting.";
    }

    if (pxbTh2PolyBarrelSummary.find(currentHistoName) != pxbTh2PolyBarrelSummary.end()) {
      pxbTh2PolyBarrelSummary[currentHistoName]->AddBin(binsSummary[id]->Clone());
    } else {
      throw cms::Exception("LocalError") << currentHistoName << " is not found in the Barrel Summary map! Aborting.";
    }
  }
}

//============================================================================
void Phase1PixelMaps::bookForwardBins(const std::string& currentHistoName) {
  auto theIndexedCorners = this->retrieveCorners(m_cornersFPIX, 3);

  for (const auto& entry : theIndexedCorners) {
    auto id = entry.first;
    auto detid = DetId(id);
    if (detid.subdetId() != PixelSubdetector::PixelEndcap)
      continue;

    int disk = m_trackerTopo.pxfDisk(detid);
    int side = m_trackerTopo.pxfSide(detid);

    unsigned mapIdx = disk + (side - 1) * 3 - 1;

    auto theVectX = entry.second.first;
    auto theVectY = entry.second.second;

    float vertX[] = {theVectX[0], theVectX[1], theVectX[2], theVectX[3]};
    float vertY[] = {theVectY[0], theVectY[1], theVectY[2], theVectY[3]};

    bins[id] = std::make_shared<TGraph>(4, vertX, vertY);
    bins[id]->SetName(TString::Format("%u", id));

    // Summary plot
    for (unsigned k = 0; k < 4; ++k) {
      vertX[k] += (float(side) - 1.5f) * 40.0f;
      vertY[k] += (disk - 1) * 35.0f;
    }

    binsSummary[id] = std::make_shared<TGraph>(4, vertX, vertY);
    binsSummary[id]->SetName(TString::Format("%u", id));

    if (pxfTh2PolyForward.find(currentHistoName) != pxfTh2PolyForward.end()) {
      pxfTh2PolyForward[currentHistoName][mapIdx]->AddBin(bins[id]->Clone());
    } else {
      throw cms::Exception("LogicError") << currentHistoName << " is not found in the Forward map! Aborting.";
    }

    if (pxfTh2PolyForwardSummary.find(currentHistoName) != pxfTh2PolyForwardSummary.end()) {
      pxfTh2PolyForwardSummary[currentHistoName]->AddBin(binsSummary[id]->Clone());
    } else {
      throw cms::Exception("LogicError") << currentHistoName << " is not found in the Forward Summary map! Aborting.";
    }
  }
}

//============================================================================
void Phase1PixelMaps::book(const std::string& currentHistoName, const char* what, const char* zaxis) {
  bookBarrelHistograms(currentHistoName, what, zaxis);
  bookForwardHistograms(currentHistoName, what, zaxis);
  m_isBooked = std::make_pair(true, true);
}

//============================================================================
void Phase1PixelMaps::fill(const std::string& currentHistoName, unsigned int id, double value) {
  auto detid = DetId(id);
  if (detid.subdetId() == PixelSubdetector::PixelBarrel) {
    int layer = m_trackerTopo.pxbLayer(id);
    if (!m_isBooked.first) {
      edm::LogError("Phase1PixelMaps") << __func__ << ": trying to fill a histogram not booked";
      return;
    }

    LogDebug("Phase1PixelMaps") << __func__ << " filling barrel with value: " << value << std::endl;

    pxbTh2PolyBarrel[currentHistoName][layer - 1]->Fill(TString::Format("%u", id), value);
    pxbTh2PolyBarrelSummary[currentHistoName]->Fill(TString::Format("%u", id), value);
  } else if (detid.subdetId() == PixelSubdetector::PixelEndcap) {
    int disk = m_trackerTopo.pxfDisk(id);
    int side = m_trackerTopo.pxfSide(id);
    unsigned mapIdx = disk + (side - 1) * 3 - 1;
    if (!m_isBooked.second) {
      edm::LogError("Phase1PixelMaps") << __func__ << ": trying to fill a histogram not booked";
      return;
    }

    LogDebug("Phase1PixelMaps") << __func__ << " filling endcaps with value: " << value << std::endl;

    pxfTh2PolyForward[currentHistoName][mapIdx]->Fill(TString::Format("%u", id), value);
    pxfTh2PolyForwardSummary[currentHistoName]->Fill(TString::Format("%u", id), value);
  }
}

//============================================================================
void Phase1PixelMaps::fillBarrelBin(const std::string& currentHistoName, unsigned int id, double value) {
  auto detid = DetId(id);
  if (detid.subdetId() != PixelSubdetector::PixelBarrel) {
    edm::LogError("Phase1PixelMaps") << "fillBarrelBin() The following detid " << id << " is not Pixel Barrel!"
                                     << std::endl;
    return;
  }
  if (!m_isBooked.first) {
    edm::LogError("Phase1PixelMaps") << __func__ << ": trying to fill a histogram not booked";
    return;
  }
  int layer = m_trackerTopo.pxbLayer(id);
  pxbTh2PolyBarrel[currentHistoName][layer - 1]->Fill(TString::Format("%u", id), value);
  pxbTh2PolyBarrelSummary[currentHistoName]->Fill(TString::Format("%u", id), value);
}

//============================================================================
void Phase1PixelMaps::fillForwardBin(const std::string& currentHistoName, unsigned int id, double value) {
  auto detid = DetId(id);
  if (detid.subdetId() != PixelSubdetector::PixelEndcap) {
    edm::LogError("Phase1PixelMaps") << "fillForwardBin() The following detid " << id << " is not Pixel Forward!"
                                     << std::endl;
    return;
  }
  if (!m_isBooked.second) {
    edm::LogError("Phase1PixelMaps") << __func__ << ": trying to fill a histogram not booked";
    return;
  }
  int disk = m_trackerTopo.pxfDisk(id);
  int side = m_trackerTopo.pxfSide(id);
  unsigned mapIdx = disk + (side - 1) * 3 - 1;
  pxfTh2PolyForward[currentHistoName][mapIdx]->Fill(TString::Format("%u", id), value);
  pxfTh2PolyForwardSummary[currentHistoName]->Fill(TString::Format("%u", id), value);
}

//============================================================================
void Phase1PixelMaps::beautifyAllHistograms() {
  if (!m_isBooked.first && !m_isBooked.second) {
    edm::LogError("Phase1PixelMaps") << __func__ << ": trying to beautify a histogram not booked";
    return;
  }

  // only if the barrel is booked
  if (m_isBooked.first) {
    for (const auto& vec : pxbTh2PolyBarrel) {
      for (const auto& plot : vec.second) {
        this->makeNicePlotStyle(plot.get());
        plot->GetXaxis()->SetTitleOffset(0.9);
        plot->GetYaxis()->SetTitleOffset(0.9);
        plot->GetZaxis()->SetTitleOffset(1.2);
        plot->GetZaxis()->SetTitleSize(0.05);
      }
    }
  }

  // only if the forwards are booked
  if (m_isBooked.second) {
    for (const auto& vec : pxfTh2PolyForward) {
      for (const auto& plot : vec.second) {
        this->makeNicePlotStyle(plot.get());
        plot->GetXaxis()->SetTitleOffset(0.9);
        plot->GetYaxis()->SetTitleOffset(0.9);
        plot->GetZaxis()->SetTitleOffset(1.2);
        plot->GetZaxis()->SetTitleSize(0.05);
      }
    }
  }
}

//============================================================================
void Phase1PixelMaps::setBarrelScale(const std::string& currentHistoName, std::pair<float, float> extrema) {
  for (auto& histo : pxbTh2PolyBarrel[currentHistoName]) {
    histo->GetZaxis()->SetRangeUser(extrema.first, extrema.second);
  }
}

//============================================================================
void Phase1PixelMaps::setForwardScale(const std::string& currentHistoName, std::pair<float, float> extrema) {
  for (auto& histo : pxfTh2PolyForward[currentHistoName]) {
    histo->GetZaxis()->SetRangeUser(extrema.first, extrema.second);
  }
}

//============================================================================
void Phase1PixelMaps::drawBarrelMaps(const std::string& currentHistoName, TCanvas& canvas, const char* drawOption) {
  auto found = (std::find(m_knownNames.begin(), m_knownNames.end(), currentHistoName) != m_knownNames.end());

  if (!m_isBooked.first || !found) {
    edm::LogError("Phase1PixelMaps") << __func__ << ": trying to draw a histogram not booked";
    return;
  }

  TPad* pad1 = new TPad("pad1", "pad1", 0.0, 0.025, 1.0, 1.0);
  TPad* pad2 = new TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.025);
  pad1->Divide(2, 2);
  pad1->Draw();
  pad2->Draw();
  for (int i = 1; i <= 4; i++) {
    pad1->cd(i);
    if (strcmp(m_option, "text") == 0) {
      pad1->cd(i)->SetRightMargin(0.02);
      pxbTh2PolyBarrel[currentHistoName].at(i - 1)->SetMarkerColor(kRed);
    } else {
      if (m_autorescale)
        rescaleAllBarrel(currentHistoName);
      adjustCanvasMargins(pad1->cd(i), 0.07, 0.12, 0.10, 0.18);
    }
    if (drawOption) {
      pxbTh2PolyBarrel[currentHistoName].at(i - 1)->Draw("L");
      pxbTh2PolyBarrel[currentHistoName].at(i - 1)->Draw(fmt::sprintf("%s%ssame", m_option, drawOption).c_str());
    } else {
      pxbTh2PolyBarrel[currentHistoName].at(i - 1)->Draw("L");
      pxbTh2PolyBarrel[currentHistoName].at(i - 1)->Draw(fmt::sprintf("%ssame", m_option).c_str());
    }
  }
}

//============================================================================
void Phase1PixelMaps::drawForwardMaps(const std::string& currentHistoName, TCanvas& canvas, const char* drawOption) {
  auto found = (std::find(m_knownNames.begin(), m_knownNames.end(), currentHistoName) != m_knownNames.end());

  if (!m_isBooked.second || !found) {
    edm::LogError("Phase1PixelMaps") << __func__ << ": trying to draw a histogram not booked";
    return;
  }

  TPad* pad1 = new TPad("pad1", "pad1", 0.0, 0.025, 1.0, 1.0);
  TPad* pad2 = new TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.025);
  pad1->Divide(3, 2);
  pad1->Draw();
  pad2->Draw();

  for (int i = 1; i <= 6; i++) {
    pad1->cd(i);
    if (strcmp(m_option, "text") == 0) {
      pad1->cd(i)->SetRightMargin(0.02);
      pxfTh2PolyForward[currentHistoName].at(i - 1)->SetMarkerColor(kRed);
    } else {
      if (m_autorescale)
        rescaleAllForward(currentHistoName);
      adjustCanvasMargins(pad1->cd(i), 0.07, 0.12, 0.10, 0.18);
    }
    if (drawOption) {
      pxfTh2PolyForward[currentHistoName].at(i - 1)->Draw("L");
      pxfTh2PolyForward[currentHistoName].at(i - 1)->Draw(fmt::sprintf("%s%ssame", m_option, drawOption).c_str());
    } else {
      pxfTh2PolyForward[currentHistoName].at(i - 1)->Draw("L");
      pxfTh2PolyForward[currentHistoName].at(i - 1)->Draw(fmt::sprintf("%ssame", m_option).c_str());
    }
  }
}

//============================================================================
void Phase1PixelMaps::drawSummaryMaps(const std::string& currentHistoName, TCanvas& canvas, const char* drawOption) {
  auto found = (std::find(m_knownNames.begin(), m_knownNames.end(), currentHistoName) != m_knownNames.end());

  if (!m_isBooked.second || !m_isBooked.first || !found) {
    edm::LogError("Phase1PixelMaps") << __func__ << ": trying to draw a histogram not booked";
    return;
  }

  TPad* pad1 = new TPad("pad1", "pad1", 0.0, 0.025, 1.0, 1.0);
  TPad* pad2 = new TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.025);
  pad1->Divide(2, 1);
  pad1->Draw();
  pad2->Draw();

  pad1->cd(1);
  std::string temp(m_option);  // create a std string
  auto isText = (temp.find("text") != std::string::npos);
  adjustCanvasMargins(pad1->cd(1), 0.07, 0.02, 0.01, isText ? 0.05 : 0.15);
  if (isText) {
    pxbTh2PolyBarrelSummary[currentHistoName]->SetMarkerColor(kRed);
    pxbTh2PolyBarrelSummary[currentHistoName]->SetMarkerSize(0.5);
  }
  pxbTh2PolyBarrelSummary[currentHistoName]->GetZaxis()->SetTitleOffset(1.4);
  pxbTh2PolyBarrelSummary[currentHistoName]->Draw("AL");
  pxbTh2PolyBarrelSummary[currentHistoName]->Draw(fmt::sprintf("%s%ssame", m_option, drawOption).c_str());

  pad1->cd(2);
  adjustCanvasMargins(pad1->cd(2), 0.07, 0.02, 0.01, isText ? 0.05 : 0.15);
  if (isText) {
    pxfTh2PolyForwardSummary[currentHistoName]->SetMarkerColor(kRed);
    pxfTh2PolyForwardSummary[currentHistoName]->SetMarkerSize(0.5);
  }
  pxfTh2PolyForwardSummary[currentHistoName]->GetZaxis()->SetTitleOffset(1.4);
  pxfTh2PolyForwardSummary[currentHistoName]->Draw("AL");
  pxfTh2PolyForwardSummary[currentHistoName]->Draw(fmt::sprintf("%s%ssame", m_option, drawOption).c_str());
}

//============================================================================
const indexedCorners Phase1PixelMaps::retrieveCorners(const std::vector<edm::FileInPath>& cornerFiles,
                                                      const unsigned int reads) {
  indexedCorners theOutMap;

  for (const auto& file : cornerFiles) {
    auto cornerFileName = file.fullPath();
    std::ifstream cornerFile(cornerFileName.c_str());
    if (!cornerFile.good()) {
      throw cms::Exception("FileError") << "Problem opening corner file: " << cornerFileName;
    }
    std::string line;
    while (std::getline(cornerFile, line)) {
      if (!line.empty()) {
        std::istringstream iss(line);
        unsigned int id;
        std::string name;
        std::vector<std::string> corners(reads, "");
        std::vector<float> xP, yP;

        iss >> id >> name;
        for (unsigned int i = 0; i < reads; ++i) {
          iss >> corners.at(i);
        }

        LOGDEBUG("Phase1PixelMaps") << id << " : ";
        for (unsigned int i = 0; i < reads; i++) {
          // remove the leading and trailing " signs in the corners list
          (corners[i]).erase(std::remove(corners[i].begin(), corners[i].end(), '"'), corners[i].end());
          LOGDEBUG("Phase1PixelMaps") << corners.at(i) << " ";
          typedef boost::tokenizer<boost::char_separator<char>> tokenizer;
          boost::char_separator<char> sep{","};
          tokenizer tok{corners.at(i), sep};
          for (const auto& t : tok | boost::adaptors::indexed(0)) {
            if (t.index() == 0) {
              xP.push_back(atof((t.value()).c_str()));
            } else if (t.index() == 1) {
              yP.push_back(atof((t.value()).c_str()));
            } else {
              edm::LogError("LogicError") << "There should not be any token with index " << t.index() << std::endl;
            }
          }
        }
        LOGDEBUG("Phase1PixelMaps") << std::endl;

        xP.push_back(xP.front());
        yP.push_back(yP.front());

        for (unsigned int i = 0; i < xP.size(); i++) {
          LOGDEBUG("Phase1PixelMaps") << "x[" << i << "]=" << xP[i] << " y[" << i << "]" << yP[i] << std::endl;
        }

        theOutMap[id] = std::make_pair(xP, yP);

      }  // if line is empty
    }    // loop on lines
  }      // loop on files
  return theOutMap;
}

//============================================================================
void Phase1PixelMaps::makeNicePlotStyle(TH1* hist) {
  hist->SetStats(kFALSE);
  hist->SetLineWidth(2);
  hist->GetXaxis()->CenterTitle(true);
  hist->GetYaxis()->CenterTitle(true);
  hist->GetXaxis()->SetTitleFont(42);
  hist->GetYaxis()->SetTitleFont(42);
  hist->GetXaxis()->SetTitleSize(0.05);
  hist->GetYaxis()->SetTitleSize(0.05);
  hist->GetXaxis()->SetTitleOffset(1.1);
  hist->GetYaxis()->SetTitleOffset(1.3);
  hist->GetXaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelFont(42);
  hist->GetYaxis()->SetLabelSize(.05);
  hist->GetXaxis()->SetLabelSize(.05);

  if (hist->InheritsFrom(TH2::Class())) {
    hist->GetZaxis()->SetLabelFont(42);
    hist->GetZaxis()->SetLabelFont(42);
    hist->GetZaxis()->SetLabelSize(.05);
    hist->GetZaxis()->SetLabelSize(.05);
  }
}

//============================================================================
void Phase1PixelMaps::adjustCanvasMargins(TVirtualPad* pad, float top, float bottom, float left, float right) {
  if (top > 0)
    pad->SetTopMargin(top);
  if (bottom > 0)
    pad->SetBottomMargin(bottom);
  if (left > 0)
    pad->SetLeftMargin(left);
  if (right > 0)
    pad->SetRightMargin(right);
}

//============================================================================
void Phase1PixelMaps::rescaleAllBarrel(const std::string& currentHistoName) {
  if (std::find(m_knownNames.begin(), m_knownNames.end(), currentHistoName) == m_knownNames.end()) {
    edm::LogError("Phase1PixelMaps") << __func__ << ": trying to manipulate a histogram not booked";
    return;
  }

  std::vector<float> maxima;
  std::transform(pxbTh2PolyBarrel[currentHistoName].begin(),
                 pxbTh2PolyBarrel[currentHistoName].end(),
                 std::back_inserter(maxima),
                 [](std::shared_ptr<TH2Poly> thp) -> float { return thp->GetMaximum(); });
  std::vector<float> minima;
  std::transform(pxbTh2PolyBarrel[currentHistoName].begin(),
                 pxbTh2PolyBarrel[currentHistoName].end(),
                 std::back_inserter(minima),
                 [](std::shared_ptr<TH2Poly> thp) -> float { return thp->GetMinimum(); });

  auto globalMax = *std::max_element(maxima.begin(), maxima.end());
  auto globalMin = *std::min_element(minima.begin(), minima.end());

  // in case the two coincide do not rescale
  if (globalMax == globalMin)
    return;

  for (auto& histo : pxbTh2PolyBarrel[currentHistoName]) {
    histo->GetZaxis()->SetRangeUser(globalMin, globalMax);
  }
}

//============================================================================
void Phase1PixelMaps::rescaleAllForward(const std::string& currentHistoName) {
  if (std::find(m_knownNames.begin(), m_knownNames.end(), currentHistoName) == m_knownNames.end()) {
    edm::LogError("Phase1PixelMaps") << __func__ << ": trying to manipulate a histogram not booked";
    return;
  }

  std::vector<float> maxima;
  std::transform(pxfTh2PolyForward[currentHistoName].begin(),
                 pxfTh2PolyForward[currentHistoName].end(),
                 std::back_inserter(maxima),
                 [](std::shared_ptr<TH2Poly> thp) -> float { return thp->GetMaximum(); });
  std::vector<float> minima;
  std::transform(pxfTh2PolyForward[currentHistoName].begin(),
                 pxfTh2PolyForward[currentHistoName].end(),
                 std::back_inserter(minima),
                 [](std::shared_ptr<TH2Poly> thp) -> float { return thp->GetMinimum(); });

  auto globalMax = *std::max_element(maxima.begin(), maxima.end());
  auto globalMin = *std::min_element(minima.begin(), minima.end());

  // in case the two coincide do not rescale
  if (globalMax == globalMin)
    return;

  for (auto& histo : pxfTh2PolyForward[currentHistoName]) {
    histo->GetZaxis()->SetRangeUser(globalMin, globalMax);
  }
}
