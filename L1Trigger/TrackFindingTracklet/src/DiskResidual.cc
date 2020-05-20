#include "L1Trigger/TrackFindingTracklet/interface/DiskResidual.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

using namespace std;
using namespace trklet;

void DiskResidual::init(const Settings* settings,
                        int disk,
                        int iphiresid,
                        int irresid,
                        int istubid,
                        double phiresid,
                        double rresid,
                        double phiresidapprox,
                        double rresidapprox,
                        double zstub,
                        double alpha,
                        FPGAWord ialpha,
                        const Stub* stubptr) {
  assert(abs(disk) >= 1);
  assert(abs(disk) <= 5);

  if (valid_ && (std::abs(iphiresid) > std::abs(fpgaphiresid_.value())))
    return;

  valid_ = true;

  disk_ = disk;

  fpgaphiresid_.set(iphiresid, settings->phiresidbits(), false, __LINE__, __FILE__);
  fpgarresid_.set(irresid, settings->rresidbits(), false, __LINE__, __FILE__);
  assert(istubid >= 0);
  unsigned int nbitsstubid = 10;
  fpgastubid_.set(istubid, nbitsstubid, true, __LINE__, __FILE__);
  assert(!fpgaphiresid_.atExtreme());

  phiresid_ = phiresid;
  rresid_ = rresid;

  phiresidapprox_ = phiresidapprox;
  rresidapprox_ = rresidapprox;

  zstub_ = zstub;
  alpha_ = alpha;
  ialpha_ = ialpha;
  stubptr_ = stubptr;
}
