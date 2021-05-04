#include "L1Trigger/TrackFindingTracklet/interface/LayerResidual.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

using namespace std;
using namespace trklet;

void LayerResidual::init(Settings const& settings,
                         int layer,
                         int iphiresid,
                         int izresid,
                         int istubid,
                         double phiresid,
                         double zresid,
                         double phiresidapprox,
                         double zresidapprox,
                         double rstub,
                         const Stub* stubptr) {
  assert(layer > 0);
  assert(layer <= N_LAYER);

  if (valid_ && (std::abs(iphiresid) > std::abs(fpgaphiresid_.value())))
    return;

  valid_ = true;

  layer_ = layer;

  fpgaphiresid_.set(iphiresid, settings.phiresidbits(), false, __LINE__, __FILE__);
  fpgazresid_.set(izresid, settings.zresidbits(), false, __LINE__, __FILE__);
  int nbitsid = 10;
  fpgastubid_.set(istubid, nbitsid, true, __LINE__, __FILE__);
  assert(!fpgaphiresid_.atExtreme());

  phiresid_ = phiresid;
  zresid_ = zresid;

  phiresidapprox_ = phiresidapprox;
  zresidapprox_ = zresidapprox;

  rstub_ = rstub;
  stubptr_ = stubptr;
}
