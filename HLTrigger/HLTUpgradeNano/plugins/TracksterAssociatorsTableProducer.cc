#include "PhysicsTools/NanoAOD/interface/AssociationMapFlatTableProducer.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"

typedef AssociationOneToOneFlatTableProducer<AssociationMapOneToOneFraction<SimCluster, CaloParticle>>
    SimClusterCaloParticleFractionFlatTableProducer;

typedef AssociationOneToManyFlatTableProducer<AssociationMapOneToManySharedEnergyScore<ticl::Trackster, ticl::Trackster>>
    TracksterTracksterEnergyScoreFlatTableProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimClusterCaloParticleFractionFlatTableProducer);
DEFINE_FWK_MODULE(TracksterTracksterEnergyScoreFlatTableProducer);
