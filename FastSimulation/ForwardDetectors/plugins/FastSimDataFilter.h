#ifndef FastSimDataFilter_h
#define FastSimDataFilter_h

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}  // namespace edm

namespace cms {

  class FastSimDataFilter : public edm::stream::EDFilter<> {
  public:
    FastSimDataFilter(const edm::ParameterSet& pset);
    ~FastSimDataFilter() override = default;

    bool filter(edm::Event&, const edm::EventSetup&) override;
    virtual void beginJob();
    virtual void endJob();

  private:
    typedef math::RhoEtaPhiVector Vector;
    const edm::EDGetTokenT<CaloTowerCollection> tokTowers_;

    const double towercut;
    int ntotal, npassed;
  };
}  // namespace cms
#endif
