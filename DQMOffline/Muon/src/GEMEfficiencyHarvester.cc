#include "DQMOffline/Muon/interface/GEMEfficiencyHarvester.h"

#include "CondFormats/GEMObjects/interface/GEMeMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TEfficiency.h"

GEMEfficiencyHarvester::GEMEfficiencyHarvester(const edm::ParameterSet& pset) {
  folder_ = pset.getUntrackedParameter<std::string>("folder");
  log_category_ = pset.getUntrackedParameter<std::string>("logCategory");
}

GEMEfficiencyHarvester::~GEMEfficiencyHarvester() {}

TProfile* GEMEfficiencyHarvester::computeEfficiency(
    const TH1F* passed, const TH1F* total, const char* name, const char* title, const double confidence_level) {
  if (not TEfficiency::CheckConsistency(*passed, *total)) {
    edm::LogError(log_category_) << "failed to pass TEfficiency::CheckConsistency. " << name << std::endl;
    return nullptr;
  }

  const TAxis* total_x = total->GetXaxis();

  TProfile* eff_profile = new TProfile(name, title, total_x->GetNbins(), total_x->GetXmin(), total_x->GetXmax());
  eff_profile->GetXaxis()->SetTitle(total_x->GetTitle());
  eff_profile->GetYaxis()->SetTitle("#epsilon");

  for (int bin = 1; bin < total->GetNbinsX(); bin++) {
    double num_passed = passed->GetBinContent(bin);
    double num_total = total->GetBinContent(bin);

    if (num_total < 1) {
      eff_profile->SetBinEntries(bin, 0);
      continue;
    }

    double efficiency = num_passed / num_total;

    double lower_bound = TEfficiency::ClopperPearson(num_total, num_passed, confidence_level, false);
    double upper_bound = TEfficiency::ClopperPearson(num_total, num_passed, confidence_level, true);

    double width = std::max(efficiency - lower_bound, upper_bound - efficiency);
    double error = std::hypot(efficiency, width);

    eff_profile->SetBinContent(bin, efficiency);
    eff_profile->SetBinError(bin, error);
    eff_profile->SetBinEntries(bin, 1);
  }

  return eff_profile;
}

TH2F* GEMEfficiencyHarvester::computeEfficiency(const TH2F* passed,
                                                const TH2F* total,
                                                const char* name,
                                                const char* title) {
  if (not TEfficiency::CheckConsistency(*passed, *total)) {
    edm::LogError(log_category_) << "failed to pass TEfficiency::CheckConsistency. " << name << std::endl;
    return nullptr;
  }

  TEfficiency eff(*passed, *total);
  TH2F* eff_hist = dynamic_cast<TH2F*>(eff.CreateHistogram());
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

void GEMEfficiencyHarvester::doEfficiency(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  const std::string efficiency_folder = folder_ + "/Efficiency/";
  ibooker.setCurrentFolder(efficiency_folder);
  igetter.setCurrentFolder(efficiency_folder);

  std::map<std::string, std::pair<const MonitorElement*, const MonitorElement*> > me_pairs;

  const std::string matched = "_matched";

  for (const std::string& name : igetter.getMEs()) {
    const std::string fullpath = efficiency_folder + name;
    const MonitorElement* me = igetter.get(fullpath);
    if (me == nullptr) {
      edm::LogError(log_category_) << "failed to get " << fullpath << std::endl;
      continue;
    }

    const bool is_matched = name.find(matched) != std::string::npos;

    std::string key = name;
    if (is_matched)
      key.erase(key.find(matched), matched.length());

    if (me_pairs.find(key) == me_pairs.end()) {
      me_pairs[key] = {nullptr, nullptr};
    }

    if (is_matched)
      me_pairs[key].first = me;
    else
      me_pairs[key].second = me;
  }

  for (auto&& [key, value] : me_pairs) {
    const auto& [me_passed, me_total] = value;
    if (me_passed == nullptr) {
      edm::LogError(log_category_) << "numerator is missing. " << key << std::endl;
    }

    if (me_total == nullptr) {
      edm::LogError(log_category_) << "denominator is missing. " << key << std::endl;
      continue;
    }

    if (me_passed->kind() != me_total->kind()) {
      edm::LogError(log_category_) << "inconsistency between kinds of passed and total" << key << std::endl;
      continue;
    }

    const std::string name = "eff_" + me_total->getName();
    const std::string title = me_passed->getTitle();

    if (me_passed->kind() == MonitorElement::Kind::TH1F) {
      TH1F* h_passed = me_passed->getTH1F();
      if (h_passed == nullptr) {
        edm::LogError(log_category_) << "failed to get TH1F from passed " << key << std::endl;
        continue;
      }
      h_passed->Sumw2();

      TH1F* h_total = me_total->getTH1F();
      if (h_total == nullptr) {
        edm::LogError(log_category_) << "failed to get TH1F from total" << key << std::endl;
        continue;
      }
      h_total->Sumw2();

      TProfile* eff = computeEfficiency(h_passed, h_total, name.c_str(), title.c_str());
      if (eff == nullptr) {
        edm::LogError(log_category_) << "failed to compute the efficiency " << key << std::endl;
        continue;
      }

      ibooker.bookProfile(name, eff);

    } else if (me_passed->kind() == MonitorElement::Kind::TH2F) {
      TH2F* h_passed = me_passed->getTH2F();
      if (h_passed == nullptr) {
        edm::LogError(log_category_) << "failed to get TH1F from passed " << key << std::endl;
        continue;
      }
      h_passed->Sumw2();

      TH2F* h_total = me_total->getTH2F();
      if (h_total == nullptr) {
        edm::LogError(log_category_) << "failed to get TH1F from total" << key << std::endl;
        continue;
      }
      h_total->Sumw2();

      TH2F* eff = computeEfficiency(h_passed, h_total, name.c_str(), title.c_str());
      if (eff == nullptr) {
        edm::LogError(log_category_) << "failed to compute the efficiency " << key << std::endl;
        continue;
      }

      ibooker.book2D(name, eff);

    } else {
      edm::LogError(log_category_) << "not implemented" << std::endl;
      continue;
    }
  }  // me_pairs
}

std::vector<std::string> GEMEfficiencyHarvester::splitString(std::string name, const std::string delimiter) {
  std::vector<std::string> tokens;
  size_t delimiter_pos;
  size_t delimiter_len = delimiter.length();
  while ((delimiter_pos = name.find("_")) != std::string::npos) {
    tokens.push_back(name.substr(0, delimiter_pos));
    name.erase(0, delimiter_pos + delimiter_len);
  }
  tokens.push_back(name);
  return tokens;
}

std::tuple<std::string, int, bool, int> GEMEfficiencyHarvester::parseResidualName(const std::string org_name,
                                                                                  const std::string prefix) {
  std::string name = org_name;

  // residual_x_ge-11_odd_ieta4 or residdual_x_ge+21_ieta3
  // residual_x: prefix
  name.erase(name.find(prefix), prefix.length());
  name.erase(name.find("_ge"), 3);

  const std::vector<std::string>&& tokens = splitString(name, "_");
  const size_t num_tokens = tokens.size();

  if ((num_tokens != 2) and (num_tokens != 3)) {
    return std::make_tuple("", -1, false, -1);
  }

  // station != 1
  std::string region_sign = tokens.front().substr(0, 1);

  TString station_str = tokens.front().substr(1, 1);
  TString ieta_str = tokens.back().substr(4, 1);
  TString superchamber_str = (num_tokens == 3) ? tokens[1] : "";

  int station = station_str.IsDigit() ? station_str.Atoi() : -1;
  int ieta = ieta_str.IsDigit() ? ieta_str.Atoi() : -1;

  bool is_odd;
  if (station == 1) {
    if (superchamber_str.EqualTo("odd"))
      is_odd = true;
    else if (superchamber_str.EqualTo("even"))
      is_odd = false;
    else
      return std::make_tuple("", -1, false, -1);
  } else {
    is_odd = false;
  }

  return std::make_tuple(region_sign, station, is_odd, ieta);
}

void GEMEfficiencyHarvester::doResolution(DQMStore::IBooker& ibooker,
                                          DQMStore::IGetter& igetter,
                                          const std::string prefix) {
  const std::string resolution_folder = folder_ + "/Resolution/";

  igetter.setCurrentFolder(resolution_folder);
  ibooker.setCurrentFolder(resolution_folder);

  std::map<std::tuple<std::string, int, bool>, std::vector<std::pair<int, TH1F*> > > res_data;

  for (const std::string& name : igetter.getMEs()) {
    if (name.find(prefix) == std::string::npos)
      continue;

    const std::string fullpath = resolution_folder + name;
    const MonitorElement* me = igetter.get(fullpath);
    if (me == nullptr) {
      edm::LogError(log_category_) << "failed to get " << fullpath << std::endl;
      continue;
    }

    TH1F* hist = me->getTH1F();
    if (hist == nullptr) {
      edm::LogError(log_category_) << "failed to get TH1F" << std::endl;
      continue;
    }

    auto&& [region_sign, station, is_odd, ieta] = parseResidualName(name, prefix);
    if (region_sign.empty() or station < 0 or ieta < 0) {
      // TODO
      continue;
    }

    const std::tuple<std::string, int, bool> key{region_sign, station, is_odd};

    if (res_data.find(key) == res_data.end()) {
      res_data.insert({key, std::vector<std::pair<int, TH1F*> >()});
      res_data[key].reserve(GEMeMap::maxEtaPartition_);
    }
    res_data[key].emplace_back(ieta, hist);
  }  // MonitorElement

  //////////////////////////////////////////////////////////////////////////////
  // NOTE
  //////////////////////////////////////////////////////////////////////////////
  for (auto [key, ieta_data] : res_data) {
    if (ieta_data.empty()) {
      continue;
    }

    TString tmp_title{ieta_data.front().second->GetTitle()};
    const TObjArray* tokens = tmp_title.Tokenize(":");
    TString title = dynamic_cast<TObjString*>(tokens->At(0))->GetString();

    auto&& [region_sign, station, is_odd] = key;
    TString&& name = TString::Format("%s_ge%s%d1", prefix.data(), region_sign.c_str(), station);
    title += TString::Format("GE %s%d/1", region_sign.c_str(), station);
    if (station == 1) {
      name += (is_odd ? "_odd" : "_even");
      title += (is_odd ? ", Odd Superchambers" : ", Even Superchambers");
    }

    TH2F* profile =
        new TH2F(name, title, GEMeMap::maxEtaPartition_, 0.5, GEMeMap::maxEtaPartition_ + 0.5, 2, -0.5, 1.5);
    auto x_axis = profile->GetXaxis();

    x_axis->SetTitle("i#eta");
    for (int ieta = 1; ieta <= GEMeMap::maxEtaPartition_; ieta++) {
      const std::string&& label = std::to_string(ieta);
      x_axis->SetBinLabel(ieta, label.c_str());
    }

    profile->GetYaxis()->SetBinLabel(1, "Mean");
    profile->GetYaxis()->SetBinLabel(2, "Std. Dev.");

    for (auto [ieta, hist] : ieta_data) {
      profile->SetBinContent(ieta, 1, hist->GetMean());
      profile->SetBinContent(ieta, 2, hist->GetStdDev());

      profile->SetBinError(ieta, 1, hist->GetMeanError());
      profile->SetBinError(ieta, 2, hist->GetStdDevError());
    }

    ibooker.book2D(name, profile);
  }
}

void GEMEfficiencyHarvester::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  doEfficiency(ibooker, igetter);
  doResolution(ibooker, igetter, "residual_phi");
}
