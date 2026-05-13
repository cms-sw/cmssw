#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
// #include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace reco;
using namespace reco::io_v1;

ostream& reco::io_v1::operator<<(std::ostream& out, const PFRecHitFraction& hit) {
  if (!out)
    return out;

  //   const reco::PFRecHit* rechit = hit.getRecHit();

  out << hit.fraction() << "x[" << hit.recHitRef()->detId() << "]";

  return out;
}
