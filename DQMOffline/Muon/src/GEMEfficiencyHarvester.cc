#include "DQMOffline/Muon/interface/GEMEfficiencyHarvester.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TEfficiency.h"

GEMEfficiencyHarvester::GEMEfficiencyHarvester(const edm::ParameterSet& pset) {
  folder_ = pset.getUntrackedParameter<std::string>("folder");
  log_category_ = "GEMEfficiencyHarvester";
}

GEMEfficiencyHarvester::~GEMEfficiencyHarvester() {}

void GEMEfficiencyHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("folder", "GEM/Efficiency/type0");
  descriptions.add("gemEfficiencyHarvesterDefault", desc);
}

TProfile* GEMEfficiencyHarvester::computeEfficiency(
    const TH1F* passed, const TH1F* total, const char* name, const char* title, const double confidence_level) {
  if (not TEfficiency::CheckConsistency(*passed, *total)) {
    edm::LogError(log_category_) << "failed to pass TEfficiency::CheckConsistency. " << name << std::endl;
    return nullptr;
  }

  const TAxis* total_x = total->GetXaxis();

  TProfile* eff_profile = new TProfile(name, title, total_x->GetNbins(), total_x->GetXmin(), total_x->GetXmax());
  eff_profile->GetXaxis()->SetTitle(total_x->GetTitle());
  eff_profile->GetYaxis()->SetTitle("Efficiency");

  for (int bin = 1; bin <= total->GetNbinsX(); bin++) {
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
      continue;
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
  while ((delimiter_pos = name.find('_')) != std::string::npos) {
    tokens.push_back(name.substr(0, delimiter_pos));
    name.erase(0, delimiter_pos + delimiter_len);
  }
  tokens.push_back(name);
  return tokens;
}

std::tuple<std::string, int, int> GEMEfficiencyHarvester::parseResidualName(const std::string org_name,
                                                                            const std::string prefix) {
  std::string name = org_name;

  // e.g. residual_rdphi_GE-11_R4 -> _GE-11_R4
  name.erase(name.find(prefix), prefix.length());

  // _GE-11_R4 -> -11_R4
  name.erase(name.find("_GE"), 3);

  // -11_R4 -> (-11, R4)
  const std::vector<std::string>&& tokens = splitString(name, "_");
  const size_t num_tokens = tokens.size();

  if (num_tokens != 2) {
    return std::make_tuple("", -1, -1);
  }

  // '-'11
  std::string region_sign = tokens.front().substr(0, 1);
  // -'1'1
  TString station_str = tokens.front().substr(1, 1);

  // R'4' or R'16'
  TString ieta_str = tokens.back().substr(1);

  const int station = station_str.IsDigit() ? station_str.Atoi() : -1;
  const int ieta = ieta_str.IsDigit() ? ieta_str.Atoi() : -1;

  return std::make_tuple(region_sign, station, ieta);
}

void GEMEfficiencyHarvester::doResolution(DQMStore::IBooker& ibooker,
                                          DQMStore::IGetter& igetter,
                                          const std::string prefix) {
  const std::string resolution_folder = folder_ + "/Resolution/";

  igetter.setCurrentFolder(resolution_folder);
  ibooker.setCurrentFolder(resolution_folder);

  // (histogram, (region_sign, station), ieta)
  std::vector<std::tuple<const TH1F*, std::pair<std::string, int>, int> > hist_vector;
  // (region_sign, station)
  std::vector<std::pair<std::string, int> > re_st_vec;
  // ieta
  std::vector<int> ieta_vec;

  for (const std::string& name : igetter.getMEs()) {
    if (name.find(prefix) == std::string::npos)
      continue;

    const std::string fullpath = resolution_folder + name;
    const MonitorElement* me = igetter.get(fullpath);
    if (me == nullptr) {
      edm::LogError(log_category_) << "failed to get " << fullpath << std::endl;
      continue;
    }

    const TH1F* hist = me->getTH1F();
    if (hist == nullptr) {
      edm::LogError(log_category_) << "failed to get TH1F" << std::endl;
      continue;
    }

    auto&& [region_sign, station, ieta] = parseResidualName(name, prefix);
    if (region_sign.empty() or station < 0 or ieta < 0) {
      edm::LogError(log_category_) << "failed to parse the name of the residual histogram: " << name << std::endl;
      continue;
    }
    std::pair<std::string, int> region_station(region_sign, station);

    hist_vector.emplace_back(hist, region_station, ieta);
    if (std::find(re_st_vec.begin(), re_st_vec.end(), region_station) == re_st_vec.end())
      re_st_vec.push_back(region_station);
    if (std::find(ieta_vec.begin(), ieta_vec.end(), ieta) == ieta_vec.end())
      ieta_vec.push_back(ieta);
  }  // MonitorElement

  if (hist_vector.empty()) {
    edm::LogError(log_category_) << "failed to find " << prefix << std::endl;
    return;
  }

  // NOTE
  // GE-2/1, GE-1/1, GE-0/1, GE+0/1, GE+1/1, GE+2/1
  auto f_sort = [](const std::pair<std::string, int>& lhs, const std::pair<std::string, int>& rhs) -> bool {
    if (lhs.first == rhs.first) {
      if (lhs.first == "-")
        return lhs.second > rhs.second;
      else
        return lhs.second < rhs.second;

    } else {
      return (lhs.first == "-");
    }
  };

  std::sort(re_st_vec.begin(), re_st_vec.end(), f_sort);
  std::sort(ieta_vec.begin(), ieta_vec.end());

  const int num_st = re_st_vec.size();
  const int num_ieta = ieta_vec.size();

  // NOTE
  TString tmp_title{std::get<0>(hist_vector.front())->GetTitle()};

  const TObjArray* tokens = tmp_title.Tokenize(":");
  const TString title_prefix = dynamic_cast<TObjString*>(tokens->At(0))->GetString();

  const TString h_mean_name = prefix + "_mean";
  const TString h_stddev_name = prefix + "_stddev";
  const TString h_skewness_name = prefix + "_skewness";

  const TString h_mean_title = title_prefix + " : Mean";
  const TString h_stddev_title = title_prefix + " : Standard Deviation";
  const TString h_skewness_title = title_prefix + " : Skewness";

  TH2F* h_mean = new TH2F(h_mean_name, h_mean_title, num_ieta, 0.5, num_ieta + 0.5, num_st, 0.5, num_st + 0.5);
  // x-axis
  h_mean->GetXaxis()->SetTitle("i#eta");
  for (unsigned int idx = 0; idx < ieta_vec.size(); idx++) {
    const int xbin = idx + 1;
    const char* label = Form("%d", ieta_vec[idx]);
    h_mean->GetXaxis()->SetBinLabel(xbin, label);
  }
  // y-axis
  for (unsigned int idx = 0; idx < re_st_vec.size(); idx++) {
    auto [region_sign, station] = re_st_vec[idx];
    const char* label = Form("GE%s%d/1", region_sign.c_str(), station);
    const int ybin = idx + 1;
    h_mean->GetYaxis()->SetBinLabel(ybin, label);
  }

  TH2F* h_stddev = dynamic_cast<TH2F*>(h_mean->Clone(h_stddev_name));
  TH2F* h_skewness = dynamic_cast<TH2F*>(h_mean->Clone(h_skewness_name));

  h_stddev->SetTitle(h_stddev_title);
  h_skewness->SetTitle(h_skewness_title);

  // NOTE
  for (auto [hist, region_station, ieta] : hist_vector) {
    const int xbin = findResolutionBin(ieta, ieta_vec);
    if (xbin < 0) {
      edm::LogError(log_category_) << "found a wrong x bin = " << xbin << std::endl;
      continue;
    }

    const int ybin = findResolutionBin(region_station, re_st_vec);
    if (ybin < 0) {
      edm::LogError(log_category_) << "found a wrong y bin = " << ybin << std::endl;
      continue;
    }

    h_mean->SetBinContent(xbin, ybin, hist->GetMean());
    h_stddev->SetBinContent(xbin, ybin, hist->GetStdDev());
    // FIXME
    // `GetSkewness` seems to returns nan when its histogram has no entry..
    const double skewness = hist->GetSkewness();
    if (not std::isnan(skewness))
      h_skewness->SetBinContent(xbin, ybin, skewness);

    h_mean->SetBinError(xbin, ybin, hist->GetMeanError());
    h_stddev->SetBinError(xbin, ybin, hist->GetStdDevError());
    h_skewness->SetBinError(xbin, ybin, hist->GetSkewness(11));
  }

  for (auto&& each : {h_mean, h_stddev, h_skewness}) {
    ibooker.book2D(each->GetName(), each);
  }
}

void GEMEfficiencyHarvester::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  doEfficiency(ibooker, igetter);
  doResolution(ibooker, igetter, "residual_rphi");
}
