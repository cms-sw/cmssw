#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/DetId/interface/DetId.h"


// from Surface
namespace{
  namespace sn{
    enum Side {positiveSide, negativeSide, onSurface};
    enum GlobalFace {outer,inner,zplus,zminus,phiplus,phiminus};
  }
}
#include<cassert>

namespace {
  void verify(DetId id,
	      sn::Side sn) {
    LocalTrajectoryParameter tp;
    PTrajectoryStateOnDet p(tp, id, ss);
    assert(p.detid()==id);
    assert(surfaceSide()=ss);
  }


}



int main() {

  DetId tracker(DetId::Tracker,2);
  DetId muon(DetId::Muon,3);

  verify(tracker,ss::positiveSide);
  verify(tracker,ss::negativeSide);
  verify(tracker,ss::onSurface);

  verify(muon,ss::positiveSide);
  verify(muon,ss::negativeSide);
  verify(muon,ss::onSurface);

  return 0;
}

