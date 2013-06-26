#include <cassert>
#include <algorithm>

#include "DataFormats/JetReco/interface/DiscretizedEnergyFlow.h"

namespace reco {

DiscretizedEnergyFlow::DiscretizedEnergyFlow(
    const double* data, const char* title,
    const double etaMin, const double etaMax, const double phiBin0Edge,
    const unsigned nEtaBins, const unsigned nPhiBins)
    : title_(title),
      etaMin_(etaMin),
      etaMax_(etaMax),
      phiBin0Edge_(phiBin0Edge),
      nEtaBins_(nEtaBins),
      nPhiBins_(nPhiBins)
{
    assert(data);
    assert(title);
    const unsigned nbins = nEtaBins*nPhiBins;
    data_.resize(nbins);
    std::copy(data, data+nbins, data_.begin());
}

}
