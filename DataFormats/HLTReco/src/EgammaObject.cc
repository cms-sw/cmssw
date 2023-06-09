#include "DataFormats/HLTReco/interface/EgammaObject.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

trigger::EgammaObject::EgammaObject(const reco::RecoEcalCandidate& ecalCand)
    : TriggerObject(ecalCand), hasPixelMatch_(false), superCluster_(ecalCand.superCluster()) {}

void trigger::EgammaObject::setSeeds(reco::ElectronSeedRefVector seeds) {
  seeds_ = std::move(seeds);
  hasPixelMatch_ = false;
  for (const auto& seed : seeds_) {
    if (!seed->hitInfo().empty()) {
      hasPixelMatch_ = true;
      break;
    }
  }
}

bool trigger::EgammaObject::hasVar(const std::string& varName) const {
  return std::binary_search(vars_.begin(), vars_.end(), varName, VarComparer());
}

float trigger::EgammaObject::var(const std::string& varName, const bool raiseExcept) const {
  //here we have a guaranteed sorted vector with unique entries
  auto varIt = std::equal_range(vars_.begin(), vars_.end(), varName, VarComparer());
  if (varIt.first != varIt.second)
    return varIt.first->second;
  else if (raiseExcept) {
    cms::Exception ex("AttributeError");
    ex << " error variable " << varName << " is not present, variables present are " << varNamesStr();
    throw ex;
  } else {
    return std::numeric_limits<float>::max();
  }
}

std::vector<std::string> trigger::EgammaObject::varNames() const {
  std::vector<std::string> names;
  names.reserve(vars_.size());
  for (const auto& var : vars_) {
    names.push_back(var.first);
  }
  return names;
}

std::string trigger::EgammaObject::varNamesStr() const {
  std::string retVal;
  auto names = varNames();
  for (const auto& name : names) {
    if (!retVal.empty())
      retVal += " ";
    retVal += name;
  }
  return retVal;
}

void trigger::EgammaObject::setVars(std::vector<std::pair<std::string, float>> vars) {
  vars_ = std::move(vars);
  std::sort(vars_.begin(), vars_.end(), [](auto& lhs, auto& rhs) { return lhs.first < rhs.first; });
}
