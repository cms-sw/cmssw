#include "DQM/GEM/interface/GEMDQMEfficiencyCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "TEfficiency.h"

GEMDQMEfficiencyCalculator::GEMDQMEfficiencyCalculator() {}

GEMDQMEfficiencyCalculator::~GEMDQMEfficiencyCalculator() {}

//
TProfile* GEMDQMEfficiencyCalculator::computeEfficiency(const TH1F* passed,
                                                        const TH1F* total,
                                                        const char* name,
                                                        const char* title) {
  if (not TEfficiency::CheckConsistency(*passed, *total)) {
    edm::LogError(kLogCategory_) << "failed to pass TEfficiency::CheckConsistency. " << name;
    return nullptr;
  }

  const TAxis* total_x = total->GetXaxis();

  TProfile* eff_profile = new TProfile(name, title, total_x->GetNbins(), total_x->GetXmin(), total_x->GetXmax());
  eff_profile->GetXaxis()->SetTitle(total_x->GetTitle());
  eff_profile->GetYaxis()->SetTitle("Efficiency");

  for (int bin = 1; bin <= total->GetNbinsX(); bin++) {
    const double num_passed = passed->GetBinContent(bin);
    const double num_total = total->GetBinContent(bin);

    if (num_total < 1) {
      eff_profile->SetBinEntries(bin, 0);
      continue;
    }

    const double efficiency = num_passed / num_total;
    const double lower_boundary = TEfficiency::ClopperPearson(num_total, num_passed, kConfidenceLevel_, false);
    const double upper_boundary = TEfficiency::ClopperPearson(num_total, num_passed, kConfidenceLevel_, true);
    const double error = std::max(efficiency - lower_boundary, upper_boundary - efficiency);
    // NOTE tprofile
    const double profile_error = std::hypot(efficiency, error);

    eff_profile->SetBinContent(bin, efficiency);
    eff_profile->SetBinError(bin, profile_error);
    eff_profile->SetBinEntries(bin, 1);
  }

  return eff_profile;
}

//
TH2F* GEMDQMEfficiencyCalculator::computeEfficiency(const TH2F* passed,
                                                    const TH2F* total,
                                                    const char* name,
                                                    const char* title) {
  if (not TEfficiency::CheckConsistency(*passed, *total)) {
    edm::LogError(kLogCategory_) << "failed to pass TEfficiency::CheckConsistency. " << name;
    return nullptr;
  }

  TEfficiency eff(*passed, *total);
  auto eff_hist = dynamic_cast<TH2F*>(eff.CreateHistogram());
  eff_hist->SetName(name);
  eff_hist->SetTitle(title);

  const TAxis* total_x = total->GetXaxis();
  TAxis* eff_hist_x = eff_hist->GetXaxis();
  eff_hist_x->SetTitle(total_x->GetTitle());
  for (int bin = 1; bin <= total->GetNbinsX(); bin++) {
    const char* label = total_x->GetBinLabel(bin);
    eff_hist_x->SetBinLabel(bin, label);
  }

  const TAxis* total_y = total->GetYaxis();
  TAxis* eff_hist_y = eff_hist->GetYaxis();
  eff_hist_y->SetTitle(total_y->GetTitle());
  for (int bin = 1; bin <= total->GetNbinsY(); bin++) {
    const char* label = total_y->GetBinLabel(bin);
    eff_hist_y->SetBinLabel(bin, label);
  }

  return eff_hist;
}

void GEMDQMEfficiencyCalculator::drawEfficiency(DQMStore::IBooker& ibooker,
                                                DQMStore::IGetter& igetter,
                                                const std::string& folder) {
  ibooker.setCurrentFolder(folder);
  igetter.setCurrentFolder(folder);

  std::map<std::string, std::pair<const MonitorElement*, const MonitorElement*> > me_pairs;

  for (const std::string& name : igetter.getMEs()) {
    const std::string fullpath = folder + "/" + name;
    const MonitorElement* me = igetter.get(fullpath);
    if (me == nullptr) {
      edm::LogError(kLogCategory_) << "failed to get " << fullpath;
      continue;
    }

    const bool is_matched = name.find(kMatchedSuffix_) != std::string::npos;

    std::string key = name;
    if (is_matched)
      key.erase(key.find(kMatchedSuffix_), kMatchedSuffix_.length());

    if (me_pairs.find(key) == me_pairs.end()) {
      me_pairs[key] = {nullptr, nullptr};
    }

    if (is_matched)
      me_pairs[key].first = me;
    else
      me_pairs[key].second = me;
  }

  for (auto& [key, value] : me_pairs) {
    const auto& [me_passed, me_total] = value;
    if (me_passed == nullptr) {
      LogDebug(kLogCategory_) << "numerator is missing. " << key;
      continue;
    }

    if (me_total == nullptr) {
      LogDebug(kLogCategory_) << "denominator is missing. " << key;
      continue;
    }

    if (me_passed->kind() != me_total->kind()) {
      edm::LogError(kLogCategory_) << "inconsistency between kinds of passed and total" << key;
      continue;
    }

    const std::string name = "eff_" + me_total->getName();
    const std::string title = me_passed->getTitle();

    if (me_passed->kind() == MonitorElement::Kind::TH1F) {
      TH1F* h_passed = me_passed->getTH1F();
      if (h_passed == nullptr) {
        edm::LogError(kLogCategory_) << "failed to get TH1F from passed " << key;
        continue;
      }
      // h_passed->Sumw2();

      TH1F* h_total = me_total->getTH1F();
      if (h_total == nullptr) {
        edm::LogError(kLogCategory_) << "failed to get TH1F from total" << key;
        continue;
      }
      // h_total->Sumw2();

      TProfile* eff = computeEfficiency(h_passed, h_total, name.c_str(), title.c_str());
      if (eff == nullptr) {
        edm::LogError(kLogCategory_) << "failed to compute the efficiency " << key;
        continue;
      }

      ibooker.bookProfile(name, eff);

    } else if (me_passed->kind() == MonitorElement::Kind::TH2F) {
      TH2F* h_passed = me_passed->getTH2F();
      if (h_passed == nullptr) {
        edm::LogError(kLogCategory_) << "failed to get TH1F from passed " << key;
        continue;
      }
      // h_passed->Sumw2();

      TH2F* h_total = me_total->getTH2F();
      if (h_total == nullptr) {
        edm::LogError(kLogCategory_) << "failed to get TH1F from total" << key;
        continue;
      }
      // h_total->Sumw2();

      TH2F* eff = computeEfficiency(h_passed, h_total, name.c_str(), title.c_str());
      if (eff == nullptr) {
        edm::LogError(kLogCategory_) << "failed to compute the efficiency " << key;
        continue;
      }

      ibooker.book2D(name, eff);

    } else {
      edm::LogError(kLogCategory_) << "not implemented";
      continue;
    }

  }  // me_pairs
}
