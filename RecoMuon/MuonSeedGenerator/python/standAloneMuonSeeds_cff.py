import FWCore.ParameterSet.Config as cms

# Geometries
# from Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi import *
# from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *

# Old stand alone muon seed producer used priod to 2-X-X
from RecoMuon.MuonSeedGenerator.ancientMuonSeed_cfi import *

# New standalone muon producer to be used in 2-X-X
from RecoMuon.MuonSeedGenerator.MuonSeed_cfi import *


mergedStandAloneMuonSeeds = cms.EDProducer("MuonSeedMerger",
                                           SeedCollections = cms.VInputTag(cms.InputTag("ancientMuonSeed"),
                                                                           cms.InputTag("MuonSeed")
                                                                           )
                                           )

#standAloneMuonSeeds = cms.Sequence(ancientMuonSeed*MuonSeed*mergedStandAloneMuonSeeds)
standAloneMuonSeedsTask = cms.Task(ancientMuonSeed)
standAloneMuonSeeds = cms.Sequence(standAloneMuonSeedsTask)
