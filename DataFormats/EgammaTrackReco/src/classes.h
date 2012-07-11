#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/CLHEP/interface/Migration.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h" 
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateSeedAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateSuperClusterAssociation.h"
#include "DataFormats/EgammaTrackReco/interface/TrackSuperClusterAssociation.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCandidateCaloClusterAssociation.h"
#include "DataFormats/EgammaTrackReco/interface/TrackCaloClusterAssociation.h"
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h" 
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h" 
#include "DataFormats/EgammaTrackReco/interface/ConversionTrack.h" 
#include "DataFormats/EgammaTrackReco/interface/ConversionTrackFwd.h" 


#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  struct dictionary {

    reco::TrackCandidateSuperClusterAssociationCollection v5;
    edm::Wrapper<edm::ValueMap<reco::SuperClusterRef> > m5;

    reco::TrackSuperClusterAssociationCollection v6;
    edm::Wrapper<edm::ValueMap<reco::SuperClusterRef> > m6;

    reco::TrackCandidateCaloClusterPtrVectorAssociation v7;
    edm::Wrapper<edm::ValueMap<reco::CaloClusterPtrVector> > m7;

    reco::TrackCandidateCaloClusterPtrAssociation v8;
    edm::Wrapper<edm::ValueMap<reco::CaloClusterPtr> > m8;

    reco::TrackCaloClusterPtrVectorAssociation v9;
    edm::Wrapper<edm::ValueMap<reco::CaloClusterPtrVector> > m9;

    reco::TrackCaloClusterPtrAssociation v10;
    edm::Wrapper<edm::ValueMap<reco::CaloClusterPtr> > m10;

    reco::ConversionTrack v11;
    edm::Wrapper<reco::ConversionTrackCollection> m11;    
    
  };
}

