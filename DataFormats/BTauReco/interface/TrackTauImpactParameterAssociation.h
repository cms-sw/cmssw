#ifndef BTauReco_TrackTauImpactParameterAssociation_h
#define BTauReco_TrackTauImpactParameterAssociation_h

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"

namespace reco {
  typedef edm::AssociationMap <
            edm::OneToValue<
              reco::TrackCollection,
              reco::TauImpactParameterTrackData
            > 
          > TrackTauImpactParameterAssociationCollection;

  typedef TrackTauImpactParameterAssociationCollection::value_type TrackTauImpactParameterAssociation;
}

#endif
