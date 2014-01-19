#ifndef RecoTauTag_ImpactParameter_TrackTools_h
#define RecoTauTag_ImpactParameter_TrackTools_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TVector3.h"
#include "RecoTauTag/ImpactParameter/interface/TrackParticle.h"
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"

namespace tauImpactParameter {

class TrackTools {
 public:
  TrackTools(){};
  virtual ~TrackTools(){};
  static TVector3 propagateToXPosition(const TrackParticle& p, double x);
  static TVector3 propagateToYPosition(const TrackParticle& p, double y);
  static TVector3 propagateToZPosition(const TrackParticle& p, double z);
  static LorentzVectorParticle lorentzParticleAtPosition(const TrackParticle& p, const TVector3& v);
};

}
#endif


