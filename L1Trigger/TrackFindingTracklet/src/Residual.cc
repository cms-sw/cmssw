#include "L1Trigger/TrackFindingTracklet/interface/Residual.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

using namespace std;
using namespace trklet;

void Residual::init(Settings const& settings,
                    unsigned int layerdisk,
                    int iphiresid,
                    int irzresid,
                    int istubid,
                    double phiresid,
                    double rzresid,
                    double phiresidapprox,
                    double rzresidapprox,
                    const Stub* stubptr) {
  assert(layerdisk < N_LAYER + N_DISK);

  if (valid_ && (std::abs(iphiresid) > std::abs(fpgaphiresid_.value())))
    return;

  valid_ = true;

  layerdisk_ = layerdisk;

  fpgaphiresid_.set(iphiresid, settings.phiresidbits(), false, __LINE__, __FILE__);
  if (layerdisk < N_LAYER) {
    fpgarzresid_.set(irzresid, settings.zresidbits(), false, __LINE__, __FILE__);
  } else {
    fpgarzresid_.set(irzresid, settings.rresidbits(), false, __LINE__, __FILE__);
  }

  int nbitsid = 10;
  fpgastubid_.set(istubid, nbitsid, true, __LINE__, __FILE__);
  assert(!fpgaphiresid_.atExtreme());

  phiresid_ = phiresid;
  rzresid_ = rzresid;

  phiresidapprox_ = phiresidapprox;
  rzresidapprox_ = rzresidapprox;

  stubptr_ = stubptr;
}
