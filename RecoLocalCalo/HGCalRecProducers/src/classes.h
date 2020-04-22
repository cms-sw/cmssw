#include "DataFormats/Common/interface/Wrapper.h"

// Add includes for your classes here
#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToGenParticleAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/VertexAssociation.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"

namespace RecoLocalCalo_HGCalRecProducers {
  struct RecoLocalCalo_HGCalRecProducers {
    // add 'dummy' Wrapper variable for each class type you put into the Event
    edm::Wrapper<std::map<DetId,HGCRecHit*>> dummy6;
  };
}  // namespace SimDataFormats_Associations
