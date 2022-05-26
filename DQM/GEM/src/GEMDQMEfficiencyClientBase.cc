#include "DQM/GEM/interface/GEMDQMEfficiencyClientBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "TEfficiency.h"
#include "TPRegexp.h"
#include <regex>

GEMDQMEfficiencyClientBase::GEMDQMEfficiencyClientBase(const edm::ParameterSet& ps)
    : kConfidenceLevel_(ps.getUntrackedParameter<double>("confidenceLevel")),
      kLogCategory_(ps.getUntrackedParameter<std::string>("logCategory")) {}

// Returns a tuple of
//   - a boolean indicating whether the parsing is successful or not
//   - name of a variable used in the efficiency monitoring
//   - GEM subdetector name like GE11-P-L1
//   - a boolean indicating whether the name is a numerator name.
std::tuple<bool, std::string, std::string, bool> GEMDQMEfficiencyClientBase::parseEfficiencySourceName(
    std::string name) {
  // NOTE This expression must be consistent with TODO
  // TODO use regex
  const bool success = TPRegexp("\\w+(?:_match)?_GE\\d1-(P|M)[0-9\\-]*").MatchB(name);
  if (not success) {
    return std::make_tuple(success, "", "", false);
  }

  const std::string numerator_pattern = "_match";
  const auto numerator_pattern_start = name.find(numerator_pattern);
  const bool is_numerator = numerator_pattern_start != std::string::npos;
  if (is_numerator) {
    // keep a delimiter between a variable name and a GEM name
    // e.g. 'muon_pt_matched_GE11-L1' --> 'muon_pt_GE11-L1'
    name.erase(numerator_pattern_start, numerator_pattern.length());
  }
  // find the position of the delimiter.
  // Because variable name can has "_", find the last one.
  // NOTE The GEM name must not contains "_"
  const unsigned long last_pos = name.find_last_of('_');

  // "muon_pt"
  const std::string var_name = name.substr(0, last_pos);

  // "GE11-L1"
  const std::string gem_name = name.substr(last_pos + 1);
  return std::make_tuple(success, var_name, gem_name, is_numerator);
}

GEMDetId GEMDQMEfficiencyClientBase::parseGEMLabel(const std::string gem_label, const std::string delimiter) {
  // GE11-P
  // GE11-P-L1
  // GE11-P-E1

  int region = 0;
  int station = 0;
  int layer = 0;
  int chamber = 0;
  int ieta = 0;

  std::vector<std::string> tokens;

  // static const?
  const std::regex re_station{"GE\\d1"};
  const std::regex re_region{"(P|M)"};
  const std::regex re_layer{"L\\d"};
  const std::regex re_chamber_layer{"\\d+L\\d"};
  const std::regex re_ieta{"E\\d+"};

  std::string::size_type last_pos = gem_label.find_first_not_of(delimiter, 0);
  std::string::size_type pos = gem_label.find_first_of(delimiter, last_pos);
  while ((pos != std::string::npos) or (last_pos != std::string::npos)) {
    const std::string token = gem_label.substr(last_pos, pos - last_pos);

    if (std::regex_match(token, re_region)) {
      region = (token == "P") ? 1 : -1;

    } else if (std::regex_match(token, re_station)) {
      station = std::stoi(token.substr(2, 1));

    } else if (std::regex_match(token, re_layer)) {
      layer = std::stoi(token.substr(1));

    } else if (std::regex_match(token, re_chamber_layer)) {
      const unsigned long layer_prefix_pos = token.find('L');
      chamber = std::stoi(token.substr(0, layer_prefix_pos));
      layer = std::stoi(token.substr(layer_prefix_pos + 1));

    } else if (std::regex_match(token, re_ieta)) {
      ieta = std::stoi(token.substr(1));

    } else {
      edm::LogError(kLogCategory_) << "unknown pattern: " << gem_label << " --> " << token;
    }
  }

  const GEMDetId id{region, 1, station, layer, chamber, ieta};
  return id;
}

std::map<std::string, GEMDQMEfficiencyClientBase::MEPair> GEMDQMEfficiencyClientBase::makeEfficiencySourcePair(
    DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, const std::string& folder, const std::string prefix) {
  ibooker.setCurrentFolder(folder);
  igetter.setCurrentFolder(folder);

  std::map<std::string, MEPair> me_pairs;

  for (const std::string& name : igetter.getMEs()) {
    // If name doesn't start with prefix
    // The default prefix is empty string.
    if (name.rfind(prefix, 0) != 0) {
      // TODO LogDebug
      continue;
    }

    const std::string fullpath = folder + "/" + name;
    const MonitorElement* me = igetter.get(fullpath);
    if (me == nullptr) {
      edm::LogError(kLogCategory_) << "failed to get " << fullpath;
      continue;
    }

    const auto [parsing_success, var_name, gem_name, is_matched] = parseEfficiencySourceName(name);
    if (not parsing_success) {
      // TODO LogDebug
      continue;
    }

    const std::string key = var_name + "_" + gem_name;

    if (me_pairs.find(key) == me_pairs.end()) {
      me_pairs[key] = {nullptr, nullptr};
    }

    if (is_matched)
      me_pairs[key].first = me;
    else
      me_pairs[key].second = me;
  }

  // remove invalid pairs
  for (auto it = me_pairs.cbegin(); it != me_pairs.cend();) {
    auto [me_numerator, me_denominator] = (*it).second;

    bool okay = true;
    if (me_numerator == nullptr) {
      okay = false;

    } else if (me_denominator == nullptr) {
      okay = false;

    } else if (me_numerator->kind() != me_denominator->kind()) {
      okay = false;
    }

    // anyways, move on to the next one
    if (okay) {
      it++;

    } else {
      it = me_pairs.erase(it);
    }
  }

  return me_pairs;
}

void GEMDQMEfficiencyClientBase::setBins(TH1F* dst_hist, const TAxis* src_axis) {
  const int nbins = src_axis->GetNbins();
  if (src_axis->IsVariableBinSize()) {
    std::vector<double> edges;
    edges.reserve(nbins + 1);

    for (int bin = 1; bin <= nbins; bin++) {
      edges.push_back(src_axis->GetBinLowEdge(bin));
    }
    edges.push_back(src_axis->GetBinUpEdge(nbins));

    dst_hist->SetBins(nbins, &edges[0]);

  } else {
    const double xlow = src_axis->GetBinLowEdge(1);
    const double xup = src_axis->GetBinUpEdge(nbins);

    dst_hist->SetBins(nbins, xlow, xup);
  }

  for (int bin = 1; bin <= nbins; bin++) {
    const TString label{src_axis->GetBinLabel(bin)};
    if (label.Length() > 0) {
      dst_hist->GetXaxis()->SetBinLabel(bin, label);
    }
  }
}

// Returns a boolean indicating whether the numerator and the denominator are
// consistent.
//
// TEfficiency::CheckConsistency raises errors and leads to an exception.
// So, the efficiency client will skip inconsitent two histograms.
// https://github.com/root-project/root/blob/v6-24-06/hist/hist/src/TEfficiency.cxx#L1494-L1512
bool GEMDQMEfficiencyClientBase::checkConsistency(const TH1& pass, const TH1& total) {
  if (pass.GetDimension() != total.GetDimension()) {
    edm::LogError(kLogCategory_) << "numerator and denominator have different dimensions: " << pass.GetName() << " & "
                                 << total.GetName();
    return false;
  }

  if (not TEfficiency::CheckBinning(pass, total)) {
    edm::LogError(kLogCategory_) << "numerator and denominator have different binning: " << pass.GetName() << " & "
                                 << total.GetName();
    return false;
  }

  if (not TEfficiency::CheckEntries(pass, total)) {
    edm::LogError(kLogCategory_) << "numerator and denominator do not have consistent bin contents " << pass.GetName()
                                 << " & " << total.GetName();
    return false;
  }

  return true;
}

// MonitorElement doesn't support TGraphAsymmErrors
TH1F* GEMDQMEfficiencyClientBase::makeEfficiency(const TH1F* h_numerator,
                                                 const TH1F* h_denominator,
                                                 const char* name,
                                                 const char* title) {
  if (h_numerator == nullptr) {
    edm::LogError(kLogCategory_) << "numerator is nullptr";
    return nullptr;
  }

  if (h_denominator == nullptr) {
    edm::LogError(kLogCategory_) << "denominator is nulpptr";
    return nullptr;
  }

  if (not checkConsistency(*h_numerator, *h_denominator)) {
    return nullptr;
  }

  if (name == nullptr) {
    name = Form("eff_%s", h_denominator->GetName());
  }

  if (title == nullptr) {
    title = h_denominator->GetTitle();
  }

  const TAxis* x_axis = h_denominator->GetXaxis();

  // create an empty TProfile for storing efficiencies and uncertainties.
  TH1F* h_eff = new TH1F();
  h_eff->SetName(name);
  h_eff->SetTitle(title);
  h_eff->GetXaxis()->SetTitle(x_axis->GetTitle());
  h_eff->GetYaxis()->SetTitle("Efficiency");
  setBins(h_eff, h_denominator->GetXaxis());

  // efficiency calculation
  const int nbins = x_axis->GetNbins();
  for (int bin = 1; bin <= nbins; bin++) {
    const double passed = h_numerator->GetBinContent(bin);
    const double total = h_denominator->GetBinContent(bin);

    if (total < 1) {
      continue;
    }

    const double efficiency = passed / total;
    const double lower_boundary = TEfficiency::ClopperPearson(total, passed, kConfidenceLevel_, false);
    const double upper_boundary = TEfficiency::ClopperPearson(total, passed, kConfidenceLevel_, true);
    const double error = std::max(efficiency - lower_boundary, upper_boundary - efficiency);

    h_eff->SetBinContent(bin, efficiency);
    h_eff->SetBinError(bin, error);
  }

  return h_eff;
}

//
TH2F* GEMDQMEfficiencyClientBase::makeEfficiency(const TH2F* h_numerator,
                                                 const TH2F* h_denominator,
                                                 const char* name,
                                                 const char* title) {
  if (h_numerator == nullptr) {
    edm::LogError(kLogCategory_) << "numerator is nullptr";
    return nullptr;
  }

  if (h_denominator == nullptr) {
    edm::LogError(kLogCategory_) << "denominator is nulpptr";
    return nullptr;
  }

  if (not checkConsistency(*h_numerator, *h_denominator)) {
    return nullptr;
  }

  if (name == nullptr) {
    name = Form("eff_%s", h_denominator->GetName());
  }

  if (title == nullptr) {
    title = h_denominator->GetTitle();
  }

  TEfficiency eff(*h_numerator, *h_denominator);
  auto h_eff = dynamic_cast<TH2F*>(eff.CreateHistogram());
  h_eff->SetName(name);
  h_eff->SetTitle(title);

  return h_eff;
}

// FIXME TH2D::ProjectionX looks buggy
TH1F* GEMDQMEfficiencyClientBase::projectHistogram(const TH2F* h_2d, const unsigned int on_which_axis) {
  if ((on_which_axis != TH1::kXaxis) and (on_which_axis != TH1::kYaxis)) {
    edm::LogError(kLogCategory_) << "invalid choice: " << on_which_axis << "."
                                 << " choose from [TH1::kXaxis (=1), TH1::kYaxis (=2)]";
    return nullptr;
  }

  const bool on_x_axis = (on_which_axis == TH1::kXaxis);

  // on which axis is the histogram projected?
  const TAxis* src_proj_axis = on_x_axis ? h_2d->GetXaxis() : h_2d->GetYaxis();
  // along which axis do the entries accumulate?
  const TAxis* src_accum_axis = on_x_axis ? h_2d->GetYaxis() : h_2d->GetXaxis();

  const TString prefix = on_x_axis ? "_proj_on_x" : "_proj_on_y";
  const TString name = h_2d->GetName() + prefix;
  const TString title = h_2d->GetTitle();

  TH1F* h_proj = new TH1F();
  h_proj->SetName(name);
  h_proj->SetTitle(title);
  h_proj->GetXaxis()->SetTitle(src_proj_axis->GetTitle());
  setBins(h_proj, src_proj_axis);

  for (int proj_bin = 1; proj_bin <= src_proj_axis->GetNbins(); proj_bin++) {
    double cumsum = 0.0;
    for (int accum_bin = 1; accum_bin <= src_accum_axis->GetNbins(); accum_bin++) {
      if (on_x_axis) {
        cumsum += h_2d->GetBinContent(proj_bin, accum_bin);
      } else {
        cumsum += h_2d->GetBinContent(accum_bin, proj_bin);
      }
    }
    h_proj->SetBinContent(proj_bin, cumsum);
  }
  h_proj->Sumw2();
  return h_proj;
}

void GEMDQMEfficiencyClientBase::bookEfficiencyAuto(DQMStore::IBooker& ibooker,
                                                    DQMStore::IGetter& igetter,
                                                    const std::string& folder) {
  const std::map<std::string, MEPair> me_pairs = makeEfficiencySourcePair(ibooker, igetter, folder);

  for (auto& [key, value] : me_pairs) {
    const auto& [me_numerator, me_denominator] = value;

    const MonitorElement::Kind me_kind = me_numerator->kind();
    if (me_kind == MonitorElement::Kind::TH1F) {
      TH1F* h_numerator = me_numerator->getTH1F();
      if (h_numerator == nullptr) {
        edm::LogError(kLogCategory_) << "failed to get TH1F from h_numerator " << key;
        continue;
      }

      TH1F* h_denominator = me_denominator->getTH1F();
      if (h_denominator == nullptr) {
        edm::LogError(kLogCategory_) << "failed to get TH1F from h_denominator" << key;
        continue;
      }

      if (TH1F* eff = makeEfficiency(h_numerator, h_denominator)) {
        ibooker.book1D(eff->GetName(), eff);

      } else {
        // makeEfficiency will report the error.
        continue;
      }

    } else if (me_kind == MonitorElement::Kind::TH2F) {
      TH2F* h_numerator = me_numerator->getTH2F();
      if (h_numerator == nullptr) {
        edm::LogError(kLogCategory_) << "failed to get TH1F from h_numerator " << key;
        continue;
      }

      TH2F* h_denominator = me_denominator->getTH2F();
      if (h_denominator == nullptr) {
        edm::LogError(kLogCategory_) << "failed to get TH1F from h_denominator" << key;
        continue;
      }

      if (TH2F* eff = makeEfficiency(h_numerator, h_denominator)) {
        ibooker.book2D(eff->GetName(), eff);

      } else {
        // makeEfficiency will report the error.
        continue;
      }

    } else {
      edm::LogError(kLogCategory_) << "got an unepxected MonitorElement::Kind "
                                   << "0x" << std::hex << static_cast<int>(me_kind);
      continue;
    }

  }  // me_pairs
}
