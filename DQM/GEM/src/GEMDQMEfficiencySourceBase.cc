#include "DQM/GEM/interface/GEMDQMEfficiencySourceBase.h"

#include "FWCore/Utilities/interface/Likely.h"

#include "TPRegexp.h"

GEMDQMEfficiencySourceBase::GEMDQMEfficiencySourceBase(const edm::ParameterSet& ps)
    : kGEMOHStatusCollectionToken_(
          consumes<GEMOHStatusCollection>(ps.getUntrackedParameter<edm::InputTag>("ohStatusTag"))),
      kGEMVFATStatusCollectionToken_(
          consumes<GEMVFATStatusCollection>(ps.getUntrackedParameter<edm::InputTag>("vfatStatusTag"))),
      kMonitorGE11_(ps.getUntrackedParameter<bool>("monitorGE11")),
      kMonitorGE21_(ps.getUntrackedParameter<bool>("monitorGE21")),
      kMonitorGE0_(ps.getUntrackedParameter<bool>("monitorGE0")),
      kMaskChamberWithError_(ps.getUntrackedParameter<bool>("maskChamberWithError")),
      kLogCategory_(ps.getUntrackedParameter<std::string>("logCategory")) {}

// NOTE GEMDQMEfficiencyClientBase::parseEfficiencySourceName
std::string GEMDQMEfficiencySourceBase::nameNumerator(const std::string& denominator) {
  const bool success = TPRegexp("\\w+_GE\\d1-(P|M)[0-9\\-]*").MatchB(denominator);
  if (not success) {
    edm::LogError(kLogCategory_) << "denominator name not understood: " << denominator;
    return std::string{};
  }

  const std::string delimiter = "_";
  const std::string::size_type delimiter_pos = denominator.find_last_of(delimiter);
  const std::string var_name = denominator.substr(0, delimiter_pos);
  const std::string gem_name = denominator.substr(delimiter_pos + 1);

  const std::string numerator = var_name + "_match" + delimiter + gem_name;
  // e.g. denominator_name = "prop_GE11-P-L1"
  // tokens = {"prop", "11-P-L1"}

  return numerator;
}

// TODO doc
dqm::impl::MonitorElement* GEMDQMEfficiencySourceBase::bookNumerator1D(DQMStore::IBooker& ibooker,
                                                                       MonitorElement* denominator) {
  if (denominator == nullptr) {
    edm::LogError(kLogCategory_) << "denominator is nullptr";
    return nullptr;
  }

  const std::string name = nameNumerator(denominator->getName());
  if (name.empty()) {
    edm::LogError(kLogCategory_) << "denominator's name is " << denominator->getName()
                                 << " but nameNumerator returns an empty string";
    return nullptr;
  }
  TH1F* hist = dynamic_cast<TH1F*>(denominator->getTH1F()->Clone(name.c_str()));
  return ibooker.book1D(name, hist);
}

// TODO doc
dqm::impl::MonitorElement* GEMDQMEfficiencySourceBase::bookNumerator2D(DQMStore::IBooker& ibooker,
                                                                       MonitorElement* denominator) {
  if (denominator == nullptr) {
    edm::LogError(kLogCategory_) << "denominator is nullptr";
    return nullptr;
  }

  const std::string name = nameNumerator(denominator->getName());
  if (name.empty()) {
    edm::LogError(kLogCategory_) << "denominator's name is " << denominator->getName()
                                 << " but nameNumerator returns an empty string";
    return nullptr;
  }

  // TODO check if getTH2F is not None
  TH2F* hist = dynamic_cast<TH2F*>(denominator->getTH2F()->Clone(name.c_str()));
  return ibooker.book2D(name, hist);
}

// TODO docs
std::tuple<bool, int, int> GEMDQMEfficiencySourceBase::getChamberRange(const GEMStation* station) {
  if (station == nullptr) {
    return std::make_tuple(false, 0, 0);
  }

  const std::vector<const GEMSuperChamber*> superchamber_vec = station->superChambers();
  if (not checkRefs(superchamber_vec)) {
    edm::LogError(kLogCategory_) << "GEMStation::superChambers";  // FIXME
    return std::make_tuple(false, 0, 0);
  }

  std::vector<int> id_vec;
  std::transform(superchamber_vec.begin(),
                 superchamber_vec.end(),
                 std::back_inserter(id_vec),
                 [](const GEMSuperChamber* superchamber) -> int { return superchamber->id().chamber(); });
  const auto [first_chamber, last_chamber] = std::minmax_element(id_vec.begin(), id_vec.end());
  if ((first_chamber == id_vec.end()) or (last_chamber == id_vec.end())) {
    edm::LogError(kLogCategory_) << "";  // TODO
    return std::make_tuple(false, 0, 0);
  }

  return std::make_tuple(true, *first_chamber, *last_chamber);
}

// TODO docs
std::tuple<bool, int, int> GEMDQMEfficiencySourceBase::getEtaPartitionRange(const GEMStation* station) {
  if (station == nullptr) {
    return std::make_tuple(false, 0, 0);
  }

  const std::vector<const GEMSuperChamber*> superchamber_vec = station->superChambers();
  if (not checkRefs(superchamber_vec)) {
    edm::LogError(kLogCategory_) << "GEMStation::superChambers";  // FIXME
    return std::make_tuple(false, 0, 0);
  }

  const std::vector<const GEMChamber*> chamber_vec = superchamber_vec.front()->chambers();
  if (not checkRefs(chamber_vec)) {
    edm::LogError(kLogCategory_) << "";  // TODO
    return std::make_tuple(false, 0, 0);
  }
  const std::vector<const GEMEtaPartition*> eta_partition_vec = chamber_vec.front()->etaPartitions();
  if (not checkRefs(eta_partition_vec)) {
    edm::LogError(kLogCategory_) << "";  // TODO
    return std::make_tuple(false, 0, 0);
  }

  std::vector<int> ieta_vec;
  std::transform(eta_partition_vec.begin(),
                 eta_partition_vec.end(),
                 std::back_inserter(ieta_vec),
                 [](const GEMEtaPartition* each) -> int { return each->id().ieta(); });
  const auto [first_ieta, last_ieta] = std::minmax_element(ieta_vec.begin(), ieta_vec.end());
  if ((first_ieta == ieta_vec.end()) or (last_ieta == ieta_vec.end())) {
    edm::LogError(kLogCategory_) << "failed to find minmax";
    return std::make_tuple(false, 0, 0);
  }

  return std::make_tuple(true, *first_ieta, *last_ieta);
}

// TODO docs
dqm::impl::MonitorElement* GEMDQMEfficiencySourceBase::bookChamber(DQMStore::IBooker& ibooker,
                                                                   const TString& name,
                                                                   const TString& title,
                                                                   const GEMStation* station) {
  if (station == nullptr) {
    edm::LogError(kLogCategory_) << "";  // TODO
    return nullptr;
  }

  auto [success, first_chamber, last_chamber] = getChamberRange(station);
  if (not success) {
    edm::LogError(kLogCategory_) << "failed to get chambers: " << station->getName();
    return nullptr;
  }

  const double xlow = first_chamber - 0.5;
  const double xup = last_chamber + 0.5;
  const int nbinsx = last_chamber - first_chamber + 1;

  MonitorElement* me = ibooker.book1D(name, title, nbinsx, xlow, xup);
  me->setAxisTitle("Chamber", 1);

  for (int chamber = first_chamber; chamber <= last_chamber; chamber++) {
    const std::string label = std::to_string(chamber);
    me->setBinLabel(chamber, label, 1);
  }

  return me;
}

// TODO docs
dqm::impl::MonitorElement* GEMDQMEfficiencySourceBase::bookChamberEtaPartition(DQMStore::IBooker& ibooker,
                                                                               const TString& name,
                                                                               const TString& title,
                                                                               const GEMStation* station) {
  if (station == nullptr) {
    edm::LogError(kLogCategory_) << "station is nullptr";
    return nullptr;
  }

  auto [chamber_success, first_chamber, last_chamber] = getChamberRange(station);
  if (not chamber_success) {
    edm::LogError(kLogCategory_) << "getChamberRange failed";
    return nullptr;
  }

  auto [ieta_success, first_ieta, last_ieta] = getEtaPartitionRange(station);
  if (not ieta_success) {
    edm::LogError(kLogCategory_) << "getEtaPartitionRange failed";
    return nullptr;
  }

  const double xlow = first_chamber - 0.5;
  const double xup = last_chamber + 0.5;
  const int nbinsx = last_chamber - first_chamber + 1;

  const double ylow = first_ieta - 0.5;
  const double yup = last_ieta + 0.5;
  const int nbinsy = last_ieta - first_ieta + 1;

  MonitorElement* me = ibooker.book2D(name, title, nbinsx, xlow, xup, nbinsy, ylow, yup);
  me->setAxisTitle("Chamber", 1);
  me->setAxisTitle("i#eta", 2);

  for (int chamber = first_chamber; chamber <= last_chamber; chamber++) {
    const std::string label = std::to_string(chamber);
    me->setBinLabel(chamber, label, 1);
  }

  for (int ieta = first_ieta; ieta <= last_ieta; ieta++) {
    const std::string label = std::to_string(ieta);
    me->setBinLabel(ieta, label, 2);
  }

  return me;
}

// TODO docs
bool GEMDQMEfficiencySourceBase::skipGEMStation(const int station) {
  bool skip = false;

  if (station == 0) {
    skip = not kMonitorGE0_;

  } else if (station == 1) {
    skip = not kMonitorGE11_;

  } else if (station == 2) {
    skip = not kMonitorGE21_;

  } else {
    edm::LogError(kLogCategory_) << "got an unexpected GEM station " << station << ". skip this station.";
    skip = true;
  }

  return skip;
}

bool GEMDQMEfficiencySourceBase::maskChamberWithError(const GEMDetId& chamber_id,
                                                      const GEMOHStatusCollection* oh_status_collection,
                                                      const GEMVFATStatusCollection* vfat_status_collection) {
  const bool mask = true;

  for (auto iter = oh_status_collection->begin(); iter != oh_status_collection->end(); iter++) {
    const auto [oh_id, range] = (*iter);
    if (chamber_id != oh_id) {
      continue;
    }

    for (auto oh_status = range.first; oh_status != range.second; oh_status++) {
      if (oh_status->isBad()) {
        // GEMOHStatus is bad. Mask this chamber.
        return mask;
      }  // isBad
    }    // range
  }      // collection

  for (auto iter = vfat_status_collection->begin(); iter != vfat_status_collection->end(); iter++) {
    const auto [vfat_id, range] = (*iter);
    if (chamber_id != vfat_id.chamberId()) {
      continue;
    }
    for (auto vfat_status = range.first; vfat_status != range.second; vfat_status++) {
      if (vfat_status->isBad()) {
        return mask;
      }
    }  // range
  }    // collection

  return not mask;
}

// TODO docs
bool GEMDQMEfficiencySourceBase::hasMEKey(const MEMap& me_map, const GEMDetId& key) {
  const bool has_key = me_map.find(key) != me_map.end();

  if UNLIKELY (not has_key) {
    const std::string hint = me_map.empty() ? "empty" : me_map.begin()->second->getName();
    edm::LogError(kLogCategory_) << "got an invalid key: " << key << ", hint=" << hint;
  }
  return has_key;
}

void GEMDQMEfficiencySourceBase::fillME(MEMap& me_map, const GEMDetId& key, const double x) {
  if (hasMEKey(me_map, key)) {
    me_map[key]->Fill(x);
  }
}

void GEMDQMEfficiencySourceBase::fillME(MEMap& me_map, const GEMDetId& key, const double x, const double y) {
  if (hasMEKey(me_map, key)) {
    me_map[key]->Fill(x, y);
  }
}

double GEMDQMEfficiencySourceBase::clampWithAxis(const double value, const TAxis* axis) {
  const double first_bin_center = axis->GetBinCenter(1);
  const double last_bin_center = axis->GetBinCenter(axis->GetNbins());
  return std::clamp(value, first_bin_center, last_bin_center);
}

// https://github.com/cms-sw/cmssw/blob/CMSSW_12_0_0_pre3/DQMOffline/L1Trigger/src/L1TFillWithinLimits.cc
void GEMDQMEfficiencySourceBase::fillMEWithinLimits(MonitorElement* me, const double x) {
  if (me == nullptr) {
    edm::LogError(kLogCategory_) << "MonitorElement is nullptr";
    return;
  }
  // FIXME assume that GEMDQMEfficiencySourceBase uses only TH1F fo 1d histograms
  const TAxis* x_axis = me->getTH1F()->GetXaxis();
  me->Fill(clampWithAxis(x, x_axis));
}

// https://github.com/cms-sw/cmssw/blob/CMSSW_12_0_0_pre3/DQMOffline/L1Trigger/src/L1TFillWithinLimits.cc
void GEMDQMEfficiencySourceBase::fillMEWithinLimits(MonitorElement* me, const double x, const double y) {
  if (me == nullptr) {
    edm::LogError(kLogCategory_) << "MonitorElement is nullptr";
    return;
  }
  // FIXME assume that GEMDQMEfficiencySourceBase uses only TH2F fo 2d histograms
  const TH2F* hist = me->getTH2F();
  const TAxis* x_axis = hist->GetXaxis();
  const TAxis* y_axis = hist->GetYaxis();

  me->Fill(clampWithAxis(x, x_axis), clampWithAxis(y, y_axis));
}

void GEMDQMEfficiencySourceBase::fillMEWithinLimits(MEMap& me_map, const GEMDetId& key, const double x) {
  if (hasMEKey(me_map, key)) {
    fillMEWithinLimits(me_map[key], x);
  }
}

void GEMDQMEfficiencySourceBase::fillMEWithinLimits(MEMap& me_map, const GEMDetId& key, const double x, const double y) {
  if (hasMEKey(me_map, key)) {
    fillMEWithinLimits(me_map[key], x, y);
  }
}
