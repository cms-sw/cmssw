/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */
#ifndef TrackTools_h
#define TrackTools_h

#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "TVector3.h"
#include "RecoTauTag/ImpactParameter/interface/TrackParticle.h"
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"

class  TrackTools {
 public:
  TrackTools(){};
  virtual ~TrackTools(){};
  static TVector3 PropogateToXPosition(TrackParticle &p,double &x);
  static TVector3 PropogateToYPosition(TrackParticle &p,double &y);
  static TVector3 PropogateToZPosition(TrackParticle &p,double &z);
  static LorentzVectorParticle LorentzParticleAtPosition(TrackParticle &p,TVector3 &v);
};
#endif


