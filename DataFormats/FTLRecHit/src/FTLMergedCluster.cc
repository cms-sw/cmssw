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
