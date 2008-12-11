import FWCore.ParameterSet.Config as cms

# Seed generator
from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff import *

# Stand alone muon track producer
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cff import *

# Global muon track producer
from RecoMuon.GlobalMuonProducer.GlobalMuonProducer_cff import *

# TeV refinement
from RecoMuon.GlobalMuonProducer.tevMuons_cfi import *

# Muon Id producer
from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *

# Muon Isolation sequence
from RecoMuon.MuonIsolationProducers.muIsolation_cff import *
muontracking = cms.Sequence(MuonSeed*standAloneMuons*globalMuons)
muontracking_with_TeVRefinement = cms.Sequence(muontracking*tevMuons)

# Muon Reconstruction
muonreco = cms.Sequence(muontracking*muonIdProducerSequence)
muonrecowith_TeVRefinemen = cms.Sequence(muontracking_with_TeVRefinement*muonIdProducerSequence)

# Muon Reconstruction plus Isolation
muonreco_plus_isolation = cms.Sequence(muonrecowith_TeVRefinemen*muIsolation)

########################################################

# Sequence for cosmic reconstruction

# Seed generator
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *

# Stand alone muon track producer
from RecoMuon.CosmicMuonProducer.cosmicMuons_cff import *

# Global muon track producer
from RecoMuon.CosmicMuonProducer.globalCosmicMuons_cff import *

# Muon Id producer
muonsFromCosmics = RecoMuon.MuonIdentification.muons_cfi.muons.clone()

muonsFromCosmics.inputCollectionLabels = ['generalTracks', 'globalCosmicMuons', 'cosmicMuons']
muonsFromCosmics.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsFromCosmics.fillIsolation = False

muoncosmicreco = cms.Sequence(CosmicMuonSeed*cosmicMuons*globalCosmicMuons*muonsFromCosmics)
