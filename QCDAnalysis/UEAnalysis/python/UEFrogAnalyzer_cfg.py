import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("TrackingTools.TrackRefitter.TracksToTrajectories_cff")
process.load("Visualisation.Frog.Frog_Analyzer_cff")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#    'file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/QCDAnalysis/UEAnalysis/test/UEAnalysisEventContentHerwigQCDPt15.root'
#    'file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/QCDAnalysis/UEAnalysis/test/UEAnalysisEventContentHerwigQCDPt30.root'
#    'file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/QCDAnalysis/UEAnalysis/test/UEAnalysisEventContentHerwigQCDPt80.root'
#    'file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/QCDAnalysis/UEAnalysis/test/UEAnalysisEventContentHerwigQCDPt170.root'
#    'file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/QCDAnalysis/UEAnalysis/test/UEAnalysisEventContentHerwigQCDPt300.root'
    'file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/QCDAnalysis/UEAnalysis/test/UEAnalysisEventContentHerwigQCDPt470.root'
    )
)

process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi")

import RecoTracker.TrackProducer.RefitterWithMaterial_cff as refitter
process.generalRefitter = refitter.TrackRefitter.clone(
    src = cms.InputTag('generalTracks'),
    TrajectoryInEvent = cms.bool(True)
    )
process.towardsRefitter = refitter.TrackRefitter.clone(
    src = cms.InputTag('towardsTracks'),
    TrajectoryInEvent = cms.bool(True)
    )
process.transverseRefitter = refitter.TrackRefitter.clone(
    src =  cms.InputTag('transverseTracks'),
    TrajectoryInEvent = cms.bool(True)
    )
process.awayRefitter = refitter.TrackRefitter.clone(
    src = cms.InputTag('awayTracks'),
    TrajectoryInEvent = cms.bool(True)
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.generalTracks = cms.EDFilter("TracksToTrajectories",
    Tracks = cms.InputTag("generalTracks"),
    TrackTransformer = cms.PSet(
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('insideOut'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyRK')
    )
)

process.standAloneMuons = cms.EDFilter("TracksToTrajectories",
    Tracks = cms.InputTag("standAloneMuons"),
    TrackTransformer = cms.PSet(
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('insideOut'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyRK')
    )
)

process.globalMuons = cms.EDFilter("TracksToTrajectories",
    Tracks = cms.InputTag("globalMuons"),
    TrackTransformer = cms.PSet(
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('insideOut'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyRK')
    )
)


# OUT
process.OUT = cms.OutputModule("PoolOutputModule",
    fileName       = cms.untracked.string('out.root'),
    outputCommands = cms.untracked.vstring(
        'keep *'
    )
)


process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")


from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *

process.genParticles = cms.EDProducer("GenParticleProducer",
    saveBarCodes = cms.untracked.bool(True),
    src = cms.InputTag("source"),
    abortOnUnknownPDGCode = cms.untracked.bool(False)
)


#process.p = cms.Path(process.genParticles * process.TrackRefitter * process.frog)
process.p = cms.Path(process.genParticles * process.generalRefitter * process.towardsRefitter * process.transverseRefitter * process.awayRefitter * process.frog)
#process.p = cms.Path(process.frog)
process.frog.OutputFile = 'UEAnalysisEventContent.vis'

#process.frog.GenParticlesProducers    = []			#Commented Lines means default value
#process.frog.SimVertexProducers       = []                     #Commented Lines means default value
#process.frog.SimTrackProducers        = []                     #Commented Lines means default value
#process.frog.SimHitProducers          = []                     #Commented Lines means default value
process.frog.SimCaloHitProducers      = []                     #Commented Lines means default value
#process.frog.SiStripClusterProducers  = []                     #Commented Lines means default value
#process.frog.EcalRecHitProducers      = []                     #Commented Lines means default value
#process.frog.HcalHBHERecHitProducers  = []                     #Commented Lines means default value
#process.frog.HcalHORecHitProducers    = []                     #Commented Lines means default value
#process.frog.HcalHFRecHitProducers    = []                     #Commented Lines means default value
#process.frog.DTSegmentProducers       = []                     #Commented Lines means default value
#process.frog.CSCSegmentProducers      = []                     #Commented Lines means default value
#process.frog.RPCHitsProducers         = []                     #Commented Lines means default value
#process.frog.CaloTowersProducers      = []                     #Commented Lines means default value
#process.frog.NIProducers              = []                     #Commented Lines means default value
#process.frog.TrackProducers           = []                     #Commented Lines means default value
process.frog.TrajectoryProducers      = ['generalRefitter','towardsRefitter','transverseRefitter','awayRefitter'] # default 'TrackRefitter'
#process.frog.BasicJetsProducers       = []                     #Commented Lines means default value
#process.frog.CaloJetsProducers        = []                     #Commented Lines means default value
#process.frog.RecoCandidateProducers   = []                     #Commented Lines means default value
#process.frog.CaloMETProducers         = []                     #Commented Lines means default value



#process.outpath  = cms.EndPath(process.OUT)
#process.schedule = cms.Schedule(process.p, process.outpath)

