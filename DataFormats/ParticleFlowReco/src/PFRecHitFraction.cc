#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
// #include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace reco;


ostream& reco::operator<<(std::ostream& out, 
		    const PFRecHitFraction& hit) {

  if(!out) return out;

  const reco::PFRecHit* rechit = hit.getRecHit();
  out<<hit.energy()<<"\t"<<(*rechit);

  return out;
}
