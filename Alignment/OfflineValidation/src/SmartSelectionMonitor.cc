#include "Alignment/OfflineValidation/interface/SmartSelectionMonitor.h"

// add new histogram
TH1 *SmartSelectionMonitor::addHistogram(TH1 *h, std::string histo) {
  if (!h->GetDefaultSumw2())
    h->Sumw2();
  if (!hasBaseHisto(histo)) {
    allMonitors_[histo] = new std::map<std::string, TH1 *>;
  }
  (*allMonitors_[histo])["all"] = h;
  return (*allMonitors_[histo])["all"];
}

TH1 *SmartSelectionMonitor::addHistogram(TH1 *h) {
  if (h == nullptr)
    return nullptr;
  return addHistogram(h, h->GetName());
}

// takes care of filling an histogram
bool SmartSelectionMonitor::fillHisto(std::string name, std::string tag, double val, double weight, bool useBinWidth) {
  TH1 *h = getHisto(name, tag);
  if (h == nullptr)
    return false;
  if (useBinWidth) {
    int ibin = h->FindBin(val);
    double width = h->GetBinWidth(ibin);
    weight /= width;
  }
  h->Fill(val, weight);
  return true;
}

bool SmartSelectionMonitor::fillHisto(
    std::string name, std::vector<std::string> tags, double val, double weight, bool useBinWidth) {
  for (unsigned int i = 0; i < tags.size(); i++) {
    fillHisto(name, tags[i], val, weight, useBinWidth);
  }
  return true;
}

bool SmartSelectionMonitor::fillHisto(
    std::string name, std::vector<std::string> tags, double val, std::vector<double> weights, bool useBinWidth) {
  for (unsigned int i = 0; i < tags.size(); i++) {
    fillHisto(name, tags[i], val, weights[i], useBinWidth);
  }
  return true;
}

// takes care of filling a 2d histogram
bool SmartSelectionMonitor::fillHisto(
    std::string name, std::string tag, double valx, double valy, double weight, bool useBinWidth) {
  TH2 *h = (TH2 *)getHisto(name, tag);
  if (h == nullptr)
    return false;
  if (useBinWidth) {
    int ibin = h->FindBin(valx, valy);
    double width = h->GetBinWidth(ibin);
    weight /= width;
  }
  h->Fill(valx, valy, weight);
  return true;
}

bool SmartSelectionMonitor::fillHisto(
    std::string name, std::vector<std::string> tags, double valx, double valy, double weight, bool useBinWidth) {
  for (unsigned int i = 0; i < tags.size(); i++) {
    fillHisto(name, tags[i], valx, valy, weight, useBinWidth);
  }
  return true;
}

bool SmartSelectionMonitor::fillHisto(std::string name,
                                      std::vector<std::string> tags,
                                      double valx,
                                      double valy,
                                      std::vector<double> weights,
                                      bool useBinWidth) {
  for (unsigned int i = 0; i < tags.size(); i++) {
    fillHisto(name, tags[i], valx, valy, weights[i], useBinWidth);
  }
  return true;
}

// takes care of filling a profile
bool SmartSelectionMonitor::fillProfile(std::string name, std::string tag, double valx, double valy, double weight) {
  TProfile *h = (TProfile *)getHisto(name, tag);
  if (h == nullptr)
    return false;
  h->Fill(valx, valy, weight);
  return true;
}

bool SmartSelectionMonitor::fillProfile(
    std::string name, std::vector<std::string> tags, double valx, double valy, double weight) {
  for (unsigned int i = 0; i < tags.size(); i++) {
    fillProfile(name, tags[i], valx, valy, weight);
  }
  return true;
}

bool SmartSelectionMonitor::fillProfile(
    std::string name, std::vector<std::string> tags, double valx, double valy, std::vector<double> weights) {
  for (unsigned int i = 0; i < tags.size(); i++) {
    fillProfile(name, tags[i], valx, valy, weights[i]);
  }
  return true;
}
