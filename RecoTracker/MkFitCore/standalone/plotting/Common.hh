#ifndef _Common_
#define _Common_

#include "TString.h"
#include "TColor.h"
#include "TStyle.h"
#include "TFile.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TMarker.h"
#include "TAxis.h"
#include "TH1F.h"
#include "TF1.h"
#include "TSystem.h"

#include <iostream>
#include <vector>
#include <algorithm>

namespace {
  void setupStyle() {
    gStyle->SetOptStat(0);
    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);
  }
};  // namespace

enum ArchEnum { SNB, KNL, SKL, LNXG, LNXS };

namespace {
  ArchEnum ARCH;
  void setupARCHEnum(const TString& arch) {
    if (arch.Contains("SNB"))
      ARCH = SNB;
    else if (arch.Contains("KNL"))
      ARCH = KNL;
    else if (arch.Contains("SKL"))
      ARCH = SKL;
    else if (arch.Contains("LNX-S"))
      ARCH = LNXS;
    else if (arch.Contains("LNX-G"))
      ARCH = LNXG;
    else {
      std::cerr << arch.Data() << " is not an allowed architecture! Exiting... " << std::endl;
      exit(1);
    }
  }
};  // namespace

struct ArchOpts {
  Int_t vumin;
  Int_t vumax;

  Int_t thmin;
  Int_t thmax;

  Double_t vutimemin;
  Double_t vutimemax;

  Double_t thtimemin;
  Double_t thtimemax;

  Double_t vuspeedupmin;
  Double_t vuspeedupmax;

  Double_t thspeedupmin;
  Double_t thspeedupmax;

  Double_t thmeiftimemin;
  Double_t thmeiftimemax;

  Double_t thmeifspeedupmin;
  Double_t thmeifspeedupmax;
};

namespace {
  ArchOpts arch_opt;
  void setupArch() {
    if (ARCH == SNB) {
      arch_opt.vumin = 1;
      arch_opt.vumax = 8;

      arch_opt.thmin = 1;
      arch_opt.thmax = 24;

      arch_opt.vutimemin = 0.;
      arch_opt.vutimemax = 0.5;

      arch_opt.thtimemin = 0.001;
      arch_opt.thtimemax = 1.;

      arch_opt.vuspeedupmin = 0.;
      arch_opt.vuspeedupmax = arch_opt.vumax;

      arch_opt.thspeedupmin = 0.;
      arch_opt.thspeedupmax = arch_opt.thmax;

      arch_opt.thmeiftimemin = 0.01;
      arch_opt.thmeiftimemax = 0.5;

      arch_opt.thmeifspeedupmin = 0.;
      arch_opt.thmeifspeedupmax = arch_opt.thmax;
    } else if (ARCH == KNL) {
      arch_opt.vumin = 1;
      arch_opt.vumax = 16;

      arch_opt.thmin = 1;
      arch_opt.thmax = 256;

      arch_opt.vutimemin = 0.;
      arch_opt.vutimemax = 1.5;

      arch_opt.thtimemin = 0.001;
      arch_opt.thtimemax = 1.;

      arch_opt.vuspeedupmin = 0.;
      arch_opt.vuspeedupmax = arch_opt.vumax;

      arch_opt.thspeedupmin = 0.;
      arch_opt.thspeedupmax = 80.;

      arch_opt.thmeiftimemin = 0.001;
      arch_opt.thmeiftimemax = arch_opt.thtimemax;

      arch_opt.thmeifspeedupmin = 0.;
      arch_opt.thmeifspeedupmax = arch_opt.thspeedupmax;
    } else if (ARCH == SKL) {
      arch_opt.vumin = 1;
      arch_opt.vumax = 16;

      arch_opt.thmin = 1;
      arch_opt.thmax = 64;

      arch_opt.vutimemin = 0.;
      arch_opt.vutimemax = 0.25;

      arch_opt.thtimemin = 0.0001;
      arch_opt.thtimemax = 1.;

      arch_opt.vuspeedupmin = 0.;
      arch_opt.vuspeedupmax = arch_opt.vumax;

      arch_opt.thspeedupmin = 0.;
      arch_opt.thspeedupmax = arch_opt.thmax / 2;

      arch_opt.thmeiftimemin = 0.001;
      arch_opt.thmeiftimemax = arch_opt.thtimemax;

      arch_opt.thmeifspeedupmin = 0.;
      arch_opt.thmeifspeedupmax = arch_opt.thspeedupmax;
    } else if (ARCH == LNXG) {
      arch_opt.vumin = 1;
      arch_opt.vumax = 16;

      arch_opt.thmin = 1;
      arch_opt.thmax = 64;

      arch_opt.vutimemin = 0.;
      arch_opt.vutimemax = 0.25;

      arch_opt.thtimemin = 0.0001;
      arch_opt.thtimemax = 1.;

      arch_opt.vuspeedupmin = 0.;
      arch_opt.vuspeedupmax = arch_opt.vumax;

      arch_opt.thspeedupmin = 0.;
      arch_opt.thspeedupmax = arch_opt.thmax / 2;

      arch_opt.thmeiftimemin = 0.001;
      arch_opt.thmeiftimemax = arch_opt.thtimemax;

      arch_opt.thmeifspeedupmin = 0.;
      arch_opt.thmeifspeedupmax = arch_opt.thspeedupmax;
    } else if (ARCH == LNXS) {
      arch_opt.vumin = 1;
      arch_opt.vumax = 16;

      arch_opt.thmin = 1;
      arch_opt.thmax = 64;

      arch_opt.vutimemin = 0.;
      arch_opt.vutimemax = 0.25;

      arch_opt.thtimemin = 0.0001;
      arch_opt.thtimemax = 1.;

      arch_opt.vuspeedupmin = 0.;
      arch_opt.vuspeedupmax = arch_opt.vumax;

      arch_opt.thspeedupmin = 0.;
      arch_opt.thspeedupmax = arch_opt.thmax / 2;

      arch_opt.thmeiftimemin = 0.001;
      arch_opt.thmeiftimemax = arch_opt.thtimemax;

      arch_opt.thmeifspeedupmin = 0.;
      arch_opt.thmeifspeedupmax = arch_opt.thspeedupmax;
    } else {
      std::cerr << "How did this happen?? You did not specify one of the allowed ARCHs!" << std::endl;
      exit(1);
    }
  }
};  // namespace

enum SuiteEnum { full, forPR, forConf, val };

namespace {
  SuiteEnum SUITE;
  void setupSUITEEnum(const TString& suite) {
    if (suite.Contains("full"))
      SUITE = full;
    else if (suite.Contains("forPR"))
      SUITE = forPR;
    else if (suite.Contains("forConf"))
      SUITE = forConf;
    else if (suite.Contains("val"))
      SUITE = val;
    else {
      std::cerr << suite.Data() << " is not an allowed validation suite! Exiting... " << std::endl;
      exit(1);
    }
  }
};  // namespace

struct BuildOpts {
  BuildOpts() {}
  BuildOpts(const TString& name, const Color_t color, const TString& label) : name(name), color(color), label(label) {}

  TString name;
  Color_t color;
  TString label;
};
typedef std::vector<BuildOpts> BOVec;
typedef std::map<TString, BuildOpts> BOMap;

namespace {
  BOVec builds;
  UInt_t nbuilds;
  void setupBuilds(const Bool_t isBenchmark, const Bool_t includeCMSSW) {
    // tmp map to fill builds vector
    BOMap buildsMap;
    buildsMap["BH"] = {"BH", kBlue, "Best Hit"};
    buildsMap["STD"] = {"STD", kGreen + 1, "Standard"};
    buildsMap["CE"] = {"CE", kRed, "Clone Engine"};
    buildsMap["FV"] = {"FV", kMagenta, "Full Vector"};
    buildsMap["CMSSW"] = {"CMSSW", kBlack, "CMSSW"};

    // KPM: Consult ./xeon_scripts/common-variables.sh to match routines to suite
    if (SUITE == full) {
      builds.emplace_back(buildsMap["BH"]);
      builds.emplace_back(buildsMap["STD"]);
      builds.emplace_back(buildsMap["CE"]);
      builds.emplace_back(buildsMap["FV"]);
    } else if (SUITE == forPR) {
      if (isBenchmark) {
        builds.emplace_back(buildsMap["BH"]);
        builds.emplace_back(buildsMap["CE"]);
      } else {
        if (!gSystem->Getenv("MKFIT_MIMI"))  // MIMI does not support STD
          builds.emplace_back(buildsMap["STD"]);
        builds.emplace_back(buildsMap["CE"]);
      }
    } else if (SUITE == forConf) {
      builds.emplace_back(buildsMap["CE"]);
      builds.back().label = "mkFit";  // change label in legend for conference
    } else if (SUITE == val) {
      if (isBenchmark) {
        std::cout << "INFO: val mode has an empty set for isBenchmark" << std::endl;
      } else {
        builds.emplace_back(buildsMap["STD"]);
        builds.emplace_back(buildsMap["CE"]);
      }
    } else {
      std::cerr << "How did this happen?? You did not specify one of the allowed SUITEs!" << std::endl;
      exit(1);
    }

    // always check for adding in CMSSW --> never true for isBenchmark
    if (includeCMSSW)
      builds.emplace_back(buildsMap["CMSSW"]);

    // set nbuilds
    nbuilds = builds.size();
  }

};  // namespace

void GetMinMaxHist(const TH1F* hist, Double_t& min, Double_t& max) {
  for (auto ibin = 1; ibin <= hist->GetNbinsX(); ibin++) {
    const auto content = hist->GetBinContent(ibin);

    if (content < min && content != 0.0)
      min = content;
    if (content > max)
      max = content;
  }
}

void SetMinMaxHist(TH1F* hist, const Double_t min, const Double_t max, const Bool_t isLogy) {
  hist->SetMinimum(isLogy ? min / 2.0 : min / 1.05);
  hist->SetMaximum(isLogy ? max * 2.0 : max * 1.05);
}

#endif
