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
muons.fillGlobalTrackQuality = True
muons.fillGlobalCosmicCompatibility = True

#Muon Id isGood flag ValueMap producer sequence
from RecoMuon.MuonIdentification.muonSelectionTypeValueMapProducer_cff import *

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
#muonreco_with_SET = cms.Sequence(muontracking_with_SET*muonsWithSET)
#run only the tracking part for SET, after that it should be merged with the main ones at some point
muonreco_with_SET = cms.Sequence(muontracking_with_SET)

# Muon Reconstruction plus Isolation
muonreco_plus_isolation = cms.Sequence(muonrecowith_TeVRefinemen*muIsolation)
muonreco_plus_isolation_plus_SET = cms.Sequence(muonrecowith_TeVRefinemen*muonreco_with_SET*muIsolation)

# .. plus muIDmaps
# this makes me wonder if we should make this a new default name (drop all _plusX)
muonreco_plus_isolation_plus_SET_plus_muIDmaps = cms.Sequence(muonreco_plus_isolation_plus_SET*muonSelectionTypeSequence)

muonrecoComplete = cms.Sequence(muonreco_plus_isolation_plus_SET*muonSelectionTypeSequence)
muonrecoComplete_minus_muIDmaps = cms.Sequence(muonreco_plus_isolation_plus_SET)
muonrecoComplete_minus_SET_minus_muIDmaps = cms.Sequence(muonrecowith_TeVRefinemen*muIsolation)

########################################################
