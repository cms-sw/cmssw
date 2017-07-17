#ifndef RecoTauTag_ImpactParameter_ParticleBuilder_h
#define RecoTauTag_ImpactParameter_ParticleBuilder_h

/* From SimpleFits Package
 * Designed an written by
 * author: Ian M. Nugent
 * Humboldt Foundations
 */

#include "RecoTauTag/ImpactParameter/interface/TrackParticle.h"
#include "RecoTauTag/ImpactParameter/interface/LorentzVectorParticle.h"
#include "TString.h"
#include "TVector3.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

namespace tauImpactParameter {

class ParticleBuilder {
 public:
  enum CMSSWPerigee{aCurv=0,aTheta,aPhi,aTip,aLip};

  ParticleBuilder(){};
  ~ParticleBuilder(){};

  static LorentzVectorParticle createLorentzVectorParticle(const reco::TransientTrack& transTrk, const reco::Vertex& V, bool fromPerigee, bool useTrackHelixPropagation);
  static TrackParticle createTrackParticle(const reco::TransientTrack& transTrk, const GlobalPoint& p, bool fromPerigee=true, bool useTrackHelixPropogation=true);
  static reco::Vertex getVertex(const LorentzVectorParticle& p);

 private:
  static TVectorT<double> convertCMSSWTrackParToSFTrackPar(const TVectorT<double>& inpar);
  static TVectorT<double> convertCMSSWTrackPerigeeToSFTrackPar(const TVectorT<double>& inpar);
};

}
#endif


