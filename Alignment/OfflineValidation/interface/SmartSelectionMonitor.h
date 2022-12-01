#ifndef ALIGNMENT_OFFLINEVALIDATION_JETHTANALYZER_SMARTSELECTIONMONITOR_H
#define ALIGNMENT_OFFLINEVALIDATION_JETHTANALYZER_SMARTSELECTIONMONITOR_H

// system include files
#include <iostream>
#include <string>
#include <map>
#include <algorithm>
#include <vector>
#include <memory>
#include <unordered_map>

// user include files
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TString.h"
#include "TROOT.h"

class SmartSelectionMonitor {
public:
  SmartSelectionMonitor() {}
  ~SmartSelectionMonitor() = default;

  typedef std::unordered_map<std::string, std::map<std::string, TH1*>*> Monitor_t;

  //short getters
  inline Monitor_t& getAllMonitors() { return allMonitors_; }

  //checks if base Histo Exist
  inline bool hasBaseHisto(std::string histo) {
    if (allMonitors_.find(histo) == allMonitors_.end())
      return false;
    return true;
  }

  //checks if tag Exist for a given histo
  inline bool hasTag(std::map<std::string, TH1*>* map, std::string tag) {
    if (map->find(tag) != map->end())
      return true;
    if (map->find("all") == map->end())
      return false;

    TH1* base = (*map)["all"];
    TString allName = base->GetName();
    TString name = tag + "_" + allName.Data();
    TH1* h = (TH1*)base->Clone(name);
    h->SetName(name);
    h->SetTitle(name);
    h->Reset("ICE");
    h->SetDirectory(gROOT);
    (*map)[tag] = h;
    return true;
  }

  //checks if tag Exist for a given histo
  inline bool hasTag(std::string histo, std::string tag) {
    if (!hasBaseHisto(histo))
      return false;
    std::map<std::string, TH1*>* map = allMonitors_[histo];
    return hasTag(map, tag);
  }

  //get histo
  inline TH1* getHisto(std::string histo, std::string tag = "all") {
    if (!hasBaseHisto(histo))
      return nullptr;
    std::map<std::string, TH1*>* map = allMonitors_[histo];
    if (!hasTag(map, tag))
      return nullptr;
    return (*map)[tag];
  }

  //write all histo
  inline void Write() {
    for (Monitor_t::iterator it = allMonitors_.begin(); it != allMonitors_.end(); it++) {
      std::map<std::string, TH1*>* map = it->second;
      bool neverFilled = true;

      for (std::map<std::string, TH1*>::iterator h = map->begin(); h != map->end(); h++) {
        if (!(h->second)) {
          printf("histo = %30s %15s IS NULL", it->first.c_str(), h->first.c_str());
          continue;
        }
        if (h->second->GetEntries() > 0)
          neverFilled = false;

        if (h->first == "all") {
          h->second->SetName(Form("%s_%s", h->first.c_str(), h->second->GetName()));
        }
        h->second->Write();
      }

      if (neverFilled) {
        printf(
            "SmartSelectionMonitor: histo = '%s' is empty for all categories, you may want to cleanup your project to "
            "remove this histogram\n",
            it->first.c_str());
      }
    }
  }

  //scale all histo by w
  inline void Scale(double w) {
    for (Monitor_t::iterator it = allMonitors_.begin(); it != allMonitors_.end(); it++) {
      std::map<std::string, TH1*>* map = it->second;
      for (std::map<std::string, TH1*>::iterator h = map->begin(); h != map->end(); h++) {
        if (!(h->second)) {
          continue;
        }
        h->second->Scale(w);
      }
    }
  }

  //takes care of filling an histogram
  bool fillHisto(std::string name, std::string tag, double valx, double weight, bool useBinWidth = false);
  bool fillHisto(std::string name, std::string tag, double valx, double valy, double weight, bool useBinWidth = false);
  bool fillProfile(std::string name, std::string tag, double valx, double valy, double weight);

  bool fillHisto(std::string name, std::vector<std::string> tags, double valx, double weight, bool useBinWidth = false);
  bool fillHisto(std::string name,
                 std::vector<std::string> tags,
                 double valx,
                 double valy,
                 double weight,
                 bool useBinWidth = false);
  bool fillProfile(std::string name, std::vector<std::string> tags, double valx, double valy, double weight);

  bool fillHisto(std::string name,
                 std::vector<std::string> tags,
                 double valx,
                 std::vector<double> weights,
                 bool useBinWidth = false);
  bool fillHisto(std::string name,
                 std::vector<std::string> tags,
                 double valx,
                 double valy,
                 std::vector<double> weights,
                 bool useBinWidth = false);
  bool fillProfile(
      std::string name, std::vector<std::string> tags, double valx, double valy, std::vector<double> weights);

  //short inits the monitor plots for a new step
  void initMonitorForStep(std::string tag);

  //short add new histogram
  TH1* addHistogram(TH1* h, std::string tag);
  TH1* addHistogram(TH1* h);

private:
  //all the selection step monitors
  Monitor_t allMonitors_;
};

#endif
