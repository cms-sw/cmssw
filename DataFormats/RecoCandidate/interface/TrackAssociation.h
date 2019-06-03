#ifndef TrackAssociation_h
#define TrackAssociation_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/View.h"

namespace reco {

  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<TrackingParticleCollection, edm::View<reco::Track>, double> >
      SimToRecoCollection;
  typedef edm::AssociationMap<
      edm::OneToManyWithQualityGeneric<edm::View<reco::Track>, TrackingParticleCollection, double> >
      RecoToSimCollection;

}  // namespace reco

#endif
