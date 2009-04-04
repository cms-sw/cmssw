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
muontracking = cms.Sequence(standAloneMuonSeeds*standAloneMuons*globalMuons)
muontracking_with_TeVRefinement = cms.Sequence(muontracking*tevMuons)

# sequence with SET standalone muons
globalSETMuons = RecoMuon.GlobalMuonProducer.GlobalMuonProducer_cff.globalMuons.clone()
globalSETMuons.MuonCollectionLabel = cms.InputTag("standAloneSETMuons","UpdatedAtVtx")
from RecoMuon.MuonSeedGenerator.SETMuonSeed_cff import *
muontracking_with_SET = cms.Sequence(SETMuonSeed*standAloneSETMuons*globalSETMuons)

# Muon Reconstruction
muonreco = cms.Sequence(muontracking*muonIdProducerSequence)
muonrecowith_TeVRefinemen = cms.Sequence(muontracking_with_TeVRefinement*muonIdProducerSequence)

muonsWithSET = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
muonsWithSET.inputCollectionLabels = ['generalTracks', 'globalSETMuons', cms.InputTag('standAloneSETMuons','UpdatedAtVtx')] 
muonsWithSET.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']   
muonreco_with_SET = cms.Sequence(muontracking_with_SET*muonsWithSET)

# Muon Reconstruction plus Isolation
muonreco_plus_isolation = cms.Sequence(muonrecowith_TeVRefinemen*muIsolation)
muonreco_plus_isolation_plus_SET = cms.Sequence(muonrecowith_TeVRefinemen*muonreco_with_SET*muIsolation)

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
