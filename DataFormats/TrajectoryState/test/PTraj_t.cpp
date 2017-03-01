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
	      sn::Side ss) {
    LocalTrajectoryParameters tp;
    PTrajectoryStateOnDet p(tp, 0., id, ss);
    assert(p.detId()==id);
    assert(p.surfaceSide()==ss);
  }
}



int main() {

  DetId tracker(DetId::Tracker,2);
  DetId muon(DetId::Muon,3);

  verify(tracker,sn::positiveSide);
  verify(tracker,sn::negativeSide);
  verify(tracker,sn::onSurface);

  verify(muon,sn::positiveSide);
  verify(muon,sn::negativeSide);
  verify(muon,sn::onSurface);

  return 0;
}

