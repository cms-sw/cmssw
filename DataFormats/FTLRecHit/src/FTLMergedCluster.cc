#include "DataFormats/FTLRecHit/interface/FTLMergedCluster.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include <iomanip>

std::ostream& operator<<(std::ostream& s, const FTLMergedCluster& clu) {
  if (clu.id().det() == DetId::Forward && clu.id().subdetId() == FastTime) {
    s << " DetId " << clu.id().rawId() << " # hits " << std::setw(2) << clu.nHits() << " : " << std::setprecision(3)
      << std::scientific << std::setw(8) << clu.energy() << " GeV, " << std::fixed << std::setw(8) << clu.time()
      << " +/- " << std::setw(8) << clu.timeError() << " ns\n";
    for (size_t nclu = 0; nclu < clu.nHits(); nclu++) {
      s << "   hit # " << std::setw(2) << nclu << " detid " << clu.hDetId(nclu).rawId() << " r/c " << std::setw(2)
        << clu.hRow(nclu) << " " << std::setw(2) << clu.hCol(nclu) << " : " << std::setprecision(3) << std::scientific
        << std::setw(8) << clu.hEnergy(nclu) << " GeV, " << std::fixed << std::setw(8) << clu.hTime(nclu) << " +/- "
        << std::setw(8) << clu.hTimeError(nclu) << " ns\n";
    }
    return s;
  } else {
    return s << "FTLMergedCluster undefined subdetector";
  }
}

bool FTLMergedCluster::operator==(FTLMergedCluster const& lh) const {
  bool isSame = nHits() == lh.nHits();
  if (isSame) {
    for (size_t index = 0; index < nHits(); index++) {
      isSame = isSame && (hUniqueId(index) == lh.hUniqueId(index));
    }
  }
  return isSame;
}

const std::vector<DetId> FTLMergedCluster::clusterIds() const {
  if (hitsDetId_.size() == 1) {
    return hitsDetId_;
  }
  auto listId = hitsDetId_;
  std::sort(listId.begin(), listId.end());
  auto it = std::unique(listId.begin(), listId.end());
  listId.resize(std::distance(listId.begin(), it));
  return listId;
}

size_t FTLMergedCluster::nClusters() const {
  if (hitsDetId_.size() == 1) {
    return 1;
  }
  std::unordered_set<DetId> seen;
  for (const auto& x : hitsDetId_) {
    seen.insert(x);
  }
  return seen.size();
}
void FTLMergedCluster::addHit(
    const DetId& id, const int row, const int col, const float time, const float timeError, const float energy) {
  hitsDetId_.push_back(id);
  uint8_t row8 = static_cast<uint8_t>(std::clamp(row, 0, 15));
  uint8_t col8 = static_cast<uint8_t>(std::clamp(col, 0, 15));
  uint8_t rowcol = static_cast<uint8_t>((row8 << krcOffset) | col8);
  hitsRowCol_.push_back(rowcol);
  hitsTime_.push_back(time);
  hitsTimeError_.push_back(timeError);
  hitsEnergy_.push_back(energy);
}
