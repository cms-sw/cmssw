#include "DataFormats/TrackerCommon/interface/ClusterSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

const std::vector<std::string> ClusterSummary::subDetNames{
    "STRIP", "TOB", "TIB", "TID", "TEC", "PIXEL", "BPIX", "FPIX"};
const std::vector<std::vector<std::string> > ClusterSummary::subDetSelections{
    {"0x1e000000-0x1A000000", "0x1e000000-0x16000000", "0x1e000000-0x18000000", "0x1e000000-0x1C000000"},
    {"0x1e000000-0x1A000000"},
    {"0x1e000000-0x16000000"},
    {"0x1e000000-0x18000000"},
    {"0x1e000000-0x1C000000"},
    {"0x1e000000-0x12000000", "0x1e000000-0x14000000"},
    {"0x1e000000-0x12000000"},
    {"0x1e000000-0x14000000"}};
const std::vector<std::string> ClusterSummary::variableNames{"NCLUSTERS", "CLUSTERSIZE", "CLUSTERCHARGE"};

ClusterSummary::ClusterSummary() : ClusterSummary(NVALIDENUMS) {}

ClusterSummary::ClusterSummary(const int nSelections)
    : modules(nSelections), nClus(nSelections), clusSize(nSelections), clusCharge(nSelections) {
  for (int i = 0; i < nSelections; ++i)
    modules[i] = i;
}

ClusterSummary& ClusterSummary::operator=(const ClusterSummary& rhs) {
  modules = rhs.modules;
  nClus = rhs.nClus;
  clusSize = rhs.clusSize;
  clusCharge = rhs.clusCharge;
  return *this;
}

// move ctor
ClusterSummary::ClusterSummary(ClusterSummary&& other) : ClusterSummary() { *this = other; }

ClusterSummary::ClusterSummary(const ClusterSummary& src)
    : modules(src.getModules()),
      nClus(src.getNClusVector()),
      clusSize(src.getClusSizeVector()),
      clusCharge(src.getClusChargeVector()) {}

int ClusterSummary::getModuleLocation(int mod, bool warn) const {
  int iM = -1;
  for (auto m : modules) {
    ++iM;
    if (m == mod)
      return iM;
  }

  if (warn)
    edm::LogWarning("NoModule") << "No information for requested module " << mod << " (" << subDetNames[mod] << ")"
                                << ". Please check in the Provenance Infomation for proper modules.";

  return -1;
}

void ClusterSummary::copyNonEmpty(const ClusterSummary& src) {
  modules.clear();
  nClus.clear();
  clusSize.clear();
  clusCharge.clear();

  const std::vector<int>& src_modules = src.getModules();
  const std::vector<int>& src_nClus = src.getNClusVector();
  const std::vector<int>& src_clusSize = src.getClusSizeVector();
  const std::vector<float>& src_clusCharge = src.getClusChargeVector();

  modules.reserve(src_modules.size());
  nClus.reserve(src_modules.size());
  clusSize.reserve(src_modules.size());
  clusCharge.reserve(src_modules.size());

  for (unsigned int iM = 0; iM < src_nClus.size(); ++iM) {
    if (src.nClus[iM] != 0) {
      modules.push_back(src_modules[iM]);
      nClus.push_back(src_nClus[iM]);
      clusSize.push_back(src_clusSize[iM]);
      clusCharge.push_back(src_clusCharge[iM]);
    }
  }
}

void ClusterSummary::reset() {
  for (unsigned int iM = 0; iM < modules.size(); ++iM) {
    nClus[iM] = 0;
    clusSize[iM] = 0;
    clusCharge[iM] = 0;
  }
}
