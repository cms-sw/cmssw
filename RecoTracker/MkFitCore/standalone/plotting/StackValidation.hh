#ifndef _StackValidation_
#define _StackValidation_

#include "Common.hh"

#include "TEfficiency.h"
#include "TGraphAsymmErrors.h"

struct RateOpts {
  RateOpts() {}
  RateOpts(const TString& dir, const TString& sORr, const TString& rate) : dir(dir), sORr(sORr), rate(rate) {}

  TString dir;
  TString sORr;  // sim or reco
  TString rate;
};
typedef std::vector<RateOpts> ROVec;

namespace {
  TString ref;
  TString refdir;
  void setupRef(const Bool_t cmsswComp) {
    ref = (cmsswComp ? "cmssw" : "sim");
    refdir = (cmsswComp ? "_cmssw" : "");
  }

  ROVec rates;
  UInt_t nrates;
  void setupRates(const Bool_t cmsswComp) {
    rates.emplace_back("efficiency", ref, "eff");
    rates.emplace_back("inefficiency", ref, "ineff_brl");
    rates.emplace_back("inefficiency", ref, "ineff_trans");
    rates.emplace_back("inefficiency", ref, "ineff_ec");
    rates.emplace_back("fakerate", "reco", "fr");
    rates.emplace_back("duplicaterate", ref, "dr");

    // set nrates after rates is set
    nrates = rates.size();
  }

  std::vector<TString> ptcuts;
  UInt_t nptcuts;
  void setupPtCuts() {
    std::vector<Float_t> tmp_ptcuts = {0.f, 0.9f, 2.f};

    for (const auto tmp_ptcut : tmp_ptcuts) {
      TString ptcut = Form("%3.1f", tmp_ptcut);
      ptcut.ReplaceAll(".", "p");
      ptcuts.emplace_back(ptcut);
    }

    // set nptcuts once ptcuts is set
    nptcuts = ptcuts.size();
  }
};  // namespace

class StackValidation {
public:
  StackValidation(const TString& label, const TString& extra, const Bool_t cmsswComp, const TString& suite);
  ~StackValidation();
  void MakeValidationStacks();
  void MakeRatioStacks(const TString& trk);
  void MakeKinematicDiffStacks(const TString& trk);
  void MakeQualityStacks(const TString& trk);

private:
  const TString label;
  const TString extra;
  const Bool_t cmsswComp;
  const TString suite;

  // legend height
  Double_t y1;
  Double_t y2;

  std::vector<TFile*> files;
};

#endif
