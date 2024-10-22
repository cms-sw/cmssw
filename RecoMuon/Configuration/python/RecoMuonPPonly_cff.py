import FWCore.ParameterSet.Config as cms

# Seed generator
from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff import *
# Stand alone muon track producer
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cff import *

# refitted stand-alone muons.
refittedStandAloneMuons = standAloneMuons.clone(
    STATrajBuilderParameters = dict(DoRefit = True)
)
#refittedStandAloneMuons.STATrajBuilderParameters.DoRefit = True
# Displaced SA muons
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
displacedMuonSeeds = CosmicMuonSeed.clone(
    ForcePointDown = False
)

displacedStandAloneMuons = standAloneMuons.clone(
    InputObjects = 'displacedMuonSeeds',
    MuonTrajectoryBuilder = 'StandAloneMuonTrajectoryBuilder',
    TrackLoaderParameters = dict(VertexConstraint = False)
)

# Global muon track producer
from RecoMuon.GlobalMuonProducer.GlobalMuonProducer_cff import *
from RecoMuon.Configuration.iterativeTkDisplaced_cff import *
displacedGlobalMuons = globalMuons.clone(
    MuonCollectionLabel = 'displacedStandAloneMuons:',
    TrackerCollectionLabel = 'displacedTracks',
    selectHighPurity = False
)

# TeV refinement
from RecoMuon.GlobalMuonProducer.tevMuons_cfi import *

# SET Muon tracking
from RecoMuon.Configuration.SETRecoMuon_cff import *

# Muon Id producer
from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *
muons1stStep.fillGlobalTrackQuality = True

# Displaced muons
displacedMuons1stStep = muons1stStep.clone(
    inputCollectionLabels = ['displacedTracks',
                             'displacedGlobalMuons',
                             'displacedStandAloneMuons'],
    inputCollectionTypes = ['inner tracks',
                            'links',
                            'outer tracks'],
    fillGlobalTrackQuality = False
)
displacedMuons1stStep.TrackExtractorPSet.Diff_r = 0.2
displacedMuons1stStep.TrackExtractorPSet.Diff_z = 0.5

displacedMuonIdProducerTask = cms.Task(displacedMuons1stStep)

#Muon Id isGood flag ValueMap producer sequence
from RecoMuon.MuonIdentification.muonSelectionTypeValueMapProducer_cff import *

# Muon Isolation sequence
from RecoMuon.MuonIsolationProducers.muIsolation_cff import *

# ---------------------------------------------------- #
################## Make the sequences ##################
# ---------------------------------------------------- #
from Configuration.Eras.Modifier_fastSim_cff import fastSim

# Muon Tracking sequence
standalonemuontrackingTask = cms.Task(standAloneMuons,
                                      refittedStandAloneMuons,
                                      displacedMuonSeeds,
                                      displacedStandAloneMuons,
                                      standAloneMuonSeedsTask)
standalonemuontracking = cms.Sequence(standalonemuontrackingTask)
# not commisoned and not relevant in FastSim (?):
fastSim.toReplaceWith(standalonemuontrackingTask,standalonemuontrackingTask.copyAndExclude([displacedMuonSeeds,displacedStandAloneMuons]))
displacedGlobalMuonTrackingTask = cms.Task(iterDisplcedTrackingTask,displacedGlobalMuons)
displacedGlobalMuonTracking = cms.Sequence(displacedGlobalMuonTrackingTask)

globalmuontrackingTask = cms.Task(globalMuons,tevMuons,displacedGlobalMuonTrackingTask)
globalmuontracking = cms.Sequence(globalmuontrackingTask)
# not commisoned and not relevant in FastSim (?):
fastSim.toReplaceWith(globalmuontrackingTask,globalmuontrackingTask.copyAndExclude([displacedGlobalMuonTrackingTask]))
muontrackingTask = cms.Task(standalonemuontrackingTask,globalmuontrackingTask)
muontracking = cms.Sequence(muontrackingTask)
# Muon Reconstruction
muonrecoTask = cms.Task(muontrackingTask,muonIdProducerTask, displacedMuonIdProducerTask)
fastSim.toReplaceWith(muonrecoTask,muonrecoTask.copyAndExclude([displacedMuonIdProducerTask]))
muonreco = cms.Sequence(muonrecoTask)
# Muon Reconstruction plus Isolation
muonreco_plus_isolationTask = cms.Task(muonrecoTask,muIsolationTask)
muonreco_plus_isolation = cms.Sequence(muonreco_plus_isolationTask)

muonrecoComplete = cms.Sequence(muonreco_plus_isolationTask,muonSelectionTypeTask)


# _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- #
# -_-_-_- Special Sequences for Iterative tracking -_-_-_- #
# -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ #

# We need to split the muon sequence above in two, to be able to run the MuonSeeding in the tracker. So muonrecoComplete will 
# be run no longer...

#from RecoMuon.MuonIdentification.earlyMuons_cfi import earlyMuons

muonGlobalRecoTask = cms.Task(globalmuontrackingTask,
                              muonIdProducerTask,
                              displacedMuonIdProducerTask,
                              muonSelectionTypeTask,
                              muIsolationTask,
                              muIsolationDisplacedTask)
muonGlobalReco = cms.Sequence(muonGlobalRecoTask)

# ... instead, the sequences will be run in the following order:
# 1st - standalonemuontracking
# 2nd - iterative tracking (def in RecoTracker config)
# 3rd - MuonIDProducer with 1&2 as input, with special replacements; the earlyMuons above. 
# 4th - MuonSeeded tracks, inside-out and outside-in
# 5th - Merging of the new TK tracks into the generalTracks collection
# 6th - Run the remnant part of the muon sequence (muonGlobalReco) 

########################################################
# not commisoned and not relevant in FastSim (?):
fastSim.toReplaceWith(muonGlobalRecoTask, muonGlobalRecoTask.copyAndExclude([muonreco_with_SET_Task,muonSelectionTypeTask,displacedMuonIdProducerTask,muIsolationDisplacedTask]))
