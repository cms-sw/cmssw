#ifndef TrackAssociation_h
#define TrackAssociation_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco{

  typedef edm::AssociationMap<edm::OneToManyWithQuality
    <TrackingParticleCollection, reco::TrackCollection, double> >
    SimToRecoCollection;  
  typedef edm::AssociationMap<edm::OneToManyWithQuality 
    <reco::TrackCollection, TrackingParticleCollection, double> >
    RecoToSimCollection;  
  
  typedef reco::SimToRecoCollection::value_type SimToRecoAssociation;
  typedef edm::Ref<reco::SimToRecoCollection> SimToRecoAssociationRef;
  typedef edm::RefProd<reco::SimToRecoCollection> SimToRecoAssociationRefProd;
  typedef edm::RefVector<reco::SimToRecoCollection> SimToRecoAssociationRefVector;

  typedef reco::RecoToSimCollection::value_type RecoToSimAssociation;
  typedef edm::Ref<reco::RecoToSimCollection> RecoToSimAssociationRef;
  typedef edm::RefProd<reco::RecoToSimCollection> RecoToSimAssociationRefProd;
  typedef edm::RefVector<reco::RecoToSimCollection> RecoToSimAssociationRefVector;
}

#endif
