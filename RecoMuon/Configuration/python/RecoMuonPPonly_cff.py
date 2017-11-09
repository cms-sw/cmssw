import FWCore.ParameterSet.Config as cms

# Seed generator
from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff import *

# Stand alone muon track producer
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cff import *

# refitted stand-alone muons.
refittedStandAloneMuons = standAloneMuons.clone()
refittedStandAloneMuons.STATrajBuilderParameters.DoRefit = True

# Displaced SA muons
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
displacedMuonSeeds = CosmicMuonSeed.clone()
displacedMuonSeeds.ForcePointDown = False

displacedStandAloneMuons = standAloneMuons.clone()
displacedStandAloneMuons.InputObjects = cms.InputTag("displacedMuonSeeds")
displacedStandAloneMuons.MuonTrajectoryBuilder = cms.string("StandAloneMuonTrajectoryBuilder")
displacedStandAloneMuons.TrackLoaderParameters.VertexConstraint = cms.bool(False) 

# Global muon track producer
from RecoMuon.GlobalMuonProducer.GlobalMuonProducer_cff import *
from RecoMuon.Configuration.iterativeTkDisplaced_cff import *
displacedGlobalMuons = globalMuons.clone()
displacedGlobalMuons.MuonCollectionLabel = cms.InputTag("displacedStandAloneMuons","")
displacedGlobalMuons.TrackerCollectionLabel = cms.InputTag("displacedTracks")

# TeV refinement
from RecoMuon.GlobalMuonProducer.tevMuons_cfi import *

# SET Muon tracking
from RecoMuon.Configuration.SETRecoMuon_cff import *

# Muon Id producer
from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *
muons1stStep.fillGlobalTrackQuality = True

#Muon Id isGood flag ValueMap producer sequence
from RecoMuon.MuonIdentification.muonSelectionTypeValueMapProducer_cff import *

# Muon Isolation sequence
from RecoMuon.MuonIsolationProducers.muIsolation_cff import *

# ---------------------------------------------------- #
################## Make the sequences ##################
# ---------------------------------------------------- #
from Configuration.Eras.Modifier_fastSim_cff import fastSim

# Muon Tracking sequence
standalonemuontracking = cms.Sequence(standAloneMuonSeeds*standAloneMuons*refittedStandAloneMuons*displacedMuonSeeds*displacedStandAloneMuons)
# not commisoned and not relevant in FastSim (?):
fastSim.toReplaceWith(standalonemuontracking,standalonemuontracking.copyAndExclude([displacedMuonSeeds,displacedStandAloneMuons]))
displacedGlobalMuonTracking = cms.Sequence(iterDisplcedTracking*displacedGlobalMuons)
globalmuontracking = cms.Sequence(globalMuons*tevMuons*displacedGlobalMuonTracking)
# not commisoned and not relevant in FastSim (?):
fastSim.toReplaceWith(globalmuontracking,globalmuontracking.copyAndExclude([displacedGlobalMuonTracking]))
muontracking = cms.Sequence(standalonemuontracking*globalmuontracking)

# Muon Reconstruction
muonreco = cms.Sequence(muontracking*muonIdProducerSequence)

# Muon Reconstruction plus Isolation
muonreco_plus_isolation = cms.Sequence(muonreco*muIsolation)

muonrecoComplete = cms.Sequence(muonreco_plus_isolation*muonSelectionTypeSequence)


# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- #
# -_-_-_- Special Sequences for Iterative tracking -_-_-_- #
# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ #

# We need to split the muon sequence above in two, to be able to run the MuonSeeding in the tracker. So muonrecoComplete will 
# be run no longer...

#from RecoMuon.MuonIdentification.earlyMuons_cfi import earlyMuons

muonGlobalReco = cms.Sequence(globalmuontracking*muonIdProducerSequence*muonSelectionTypeSequence*muIsolation)

# ... instead, the sequences will be run in the following order:
# 1st - standalonemuontracking
# 2nd - iterative tracking (def in RecoTracker config)
# 3rd - MuonIDProducer with 1&2 as input, with special replacements; the earlyMuons above. 
# 4th - MuonSeeded tracks, inside-out and outside-in
# 5th - Merging of the new TK tracks into the generalTracks collection
# 6th - Run the remnant part of the muon sequence (muonGlobalReco) 

########################################################

from RecoMuon.MuonIdentification.me0MuonReco_cff import *
_phase2_muonGlobalReco = muonGlobalReco.copy()
_phase2_muonGlobalReco += me0MuonReco
phase2_muon.toReplaceWith( muonGlobalReco, _phase2_muonGlobalReco )

# not commisoned and not relevant in FastSim (?):
fastSim.toReplaceWith(muonGlobalReco, muonGlobalReco.copyAndExclude([muonreco_with_SET,muonSelectionTypeSequence]))
