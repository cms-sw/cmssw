import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process("DEMO",eras.Run2_2016,eras.fastSim)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# load particle data table
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
# load geometry
process.load('FastSimulation.Configuration.Geometries_MC_cff')
# load magnetic field
process.load('Configuration.StandardSequences.MagneticField_cff')
#process.load("Configuration.StandardSequences.MagneticField_0T_cff") 
#load and set conditions (required by geometry and magnetic field)
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')          

# read generator event from file
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('file:gen_muGun.root'),
    inputCommands = cms.untracked.vstring('keep *', 
        'drop *_genParticlesForJets_*_*', 
        'drop *_kt4GenJets_*_*', 
        'drop *_kt6GenJets_*_*', 
        'drop *_iterativeCone5GenJets_*_*', 
        'drop *_ak4GenJets_*_*', 
        'drop *_ak7GenJets_*_*', 
        'drop *_ak8GenJets_*_*', 
        'drop *_ak4GenJetsNoNu_*_*', 
        'drop *_ak8GenJetsNoNu_*_*', 
        'drop *_genCandidatesForMET_*_*', 
        'drop *_genParticlesForMETAllVisible_*_*', 
        'drop *_genMetCalo_*_*', 
        'drop *_genMetCaloAndNonPrompt_*_*', 
        'drop *_genMetTrue_*_*', 
        'drop *_genMetIC5GenJs_*_*'),
    secondaryFileNames = cms.untracked.vstring()
)

# configure random number generator for simhit production
process.load('Configuration.StandardSequences.Services_cff')

# How can I append the fastSimProducer = cms.PSet(...) to process.RandomNumberGeneratorService instead of copying everything?
process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    externalLHEProducer = cms.PSet(process.RandomNumberGeneratorService.externalLHEProducer),
    generator = cms.PSet(process.RandomNumberGeneratorService.generator),
    VtxSmeared = cms.PSet(process.RandomNumberGeneratorService.VtxSmeared),
    LHCTransport = cms.PSet(process.RandomNumberGeneratorService.LHCTransport),
    hiSignalLHCTransport = cms.PSet(process.RandomNumberGeneratorService.hiSignalLHCTransport),
    g4SimHits = cms.PSet(process.RandomNumberGeneratorService.g4SimHits),
    mix = cms.PSet(process.RandomNumberGeneratorService.mix),
    mixData = cms.PSet(process.RandomNumberGeneratorService.mixData),
    simSiStripDigiSimLink = cms.PSet(process.RandomNumberGeneratorService.simSiStripDigiSimLink),
    simMuonCSCDigis = cms.PSet(process.RandomNumberGeneratorService.simMuonCSCDigis),
    simMuonRPCDigis = cms.PSet(process.RandomNumberGeneratorService.simMuonRPCDigis),
    simMuonDTDigis = cms.PSet(process.RandomNumberGeneratorService.simMuonDTDigis),
    hiSignal = cms.PSet(process.RandomNumberGeneratorService.hiSignal),
    hiSignalG4SimHits = cms.PSet(process.RandomNumberGeneratorService.hiSignalG4SimHits),
    famosPileUp = cms.PSet(process.RandomNumberGeneratorService.famosPileUp),
    mixGenPU = cms.PSet(process.RandomNumberGeneratorService.mixGenPU),
    mixSimCaloHits = cms.PSet(process.RandomNumberGeneratorService.mixSimCaloHits),
    mixRecoTracks = cms.PSet(process.RandomNumberGeneratorService.mixRecoTracks),
    famosSimHits = cms.PSet(process.RandomNumberGeneratorService.famosSimHits),
    fastTrackerRecHits = cms.PSet(process.RandomNumberGeneratorService.fastTrackerRecHits),
    ecalRecHit = cms.PSet(process.RandomNumberGeneratorService.ecalRecHit),
    ecalPreshowerRecHit = cms.PSet(process.RandomNumberGeneratorService.ecalPreshowerRecHit),
    hbhereco = cms.PSet(process.RandomNumberGeneratorService.hbhereco),
    horeco = cms.PSet(process.RandomNumberGeneratorService.horeco),
    hfreco = cms.PSet(process.RandomNumberGeneratorService.hfreco),
    paramMuons = cms.PSet(process.RandomNumberGeneratorService.paramMuons),
    l1ParamMuons = cms.PSet(process.RandomNumberGeneratorService.l1ParamMuons),
    MuonSimHits = cms.PSet(process.RandomNumberGeneratorService.MuonSimHits),
    simBeamSpotFilter = cms.PSet(process.RandomNumberGeneratorService.simBeamSpotFilter),
    fastSimProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(234567),
        engineName = cms.untracked.string('TRandom3')
    ),
    saveFileName = cms.untracked.string('')
    )

# Remaining Steps
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('FastSimulation.Configuration.SimIdeal_cff')
process.load('FastSimulation.Configuration.Reconstruction_BefMix_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('FastSimulation.Configuration.Reconstruction_AftMix_cff')
process.load('Configuration.StandardSequences.Validation_cff')


# use new TrackerSimHitProducer
process.fastTrackerRecHits.simHits = cms.InputTag("fastSimProducer","TrackerHits")
process.fastMatchedTrackerRecHits.simHits = cms.InputTag("fastSimProducer","TrackerHits")
process.fastMatchedTrackerRecHitCombinations.simHits = cms.InputTag("fastSimProducer","TrackerHits")
process.simMuonCSCDigis.InputCollection = cms.string("fastSimProducerMuonCSCHits")

process.theMixObjects.mixCH.input = cms.VInputTag(cms.InputTag("fastSimProducer","EcalHitsEB"), cms.InputTag("fastSimProducer","EcalHitsEE"), cms.InputTag("fastSimProducer","EcalHitsES"), cms.InputTag("fastSimProducer","HcalHits"))
process.theMixObjects.mixSH.input = cms.VInputTag(cms.InputTag("fastSimProducer","MuonCSCHits"), cms.InputTag("fastSimProducer","MuonDTHits"), cms.InputTag("fastSimProducer","MuonRPCHits"), cms.InputTag("fastSimProducer","TrackerHits"))
process.theMixObjects.mixTracks.input = cms.VInputTag(cms.InputTag("fastSimProducer"))
process.theMixObjects.mixVertices.input = cms.VInputTag(cms.InputTag("fastSimProducer"))
process.mixCollectionValidation.theMixObjects = cms.PSet(process.theMixObjects)

process.mixSimHits.input = cms.VInputTag(cms.InputTag("fastSimProducer","MuonCSCHits"), cms.InputTag("fastSimProducer","MuonDTHits"), cms.InputTag("fastSimProducer","MuonRPCHits"), cms.InputTag("fastSimProducer","TrackerHits"))

process.theDigitizersValid.mergedtruth.simHitCollections = cms.PSet(
        muon = cms.VInputTag(cms.InputTag("fastSimProducer","MuonDTHits"), cms.InputTag("fastSimProducer","MuonCSCHits"), cms.InputTag("fastSimProducer","MuonRPCHits")),
        trackerAndPixel = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
    )
process.theDigitizersValid.mergedtruth.simTrackCollection = cms.InputTag("fastSimProducer")
process.theDigitizersValid.mergedtruth.simVertexCollection = cms.InputTag("fastSimProducer")
process.theDigitizersValid.ecal.hitsProducer = cms.string('fastSimProducer')
process.theDigitizersValid.hcal.hitsProducer = cms.string('fastSimProducer')

process.mix.hitsProducer = cms.string('fastSimProducer')
process.mix.mixObjects  = cms.PSet(process.theMixObjects)
process.mix.digitizers = cms.PSet(process.theDigitizersValid)

process.trackingParticles.simHitCollections = cms.PSet(
        muon = cms.VInputTag(cms.InputTag("fastSimProducer","MuonDTHits"), cms.InputTag("fastSimProducer","MuonCSCHits"), cms.InputTag("fastSimProducer","MuonRPCHits")),
        trackerAndPixel = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
    )
process.trackingParticles.simHitCollections.simTrackCollection = cms.InputTag("fastSimProducer")
process.trackingParticles.simHitCollections.simVertexCollection = cms.InputTag("fastSimProducer")

process.simHitTPAssocProducer.simHitSrc = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"), cms.InputTag("fastSimProducer","MuonCSCHits"), cms.InputTag("fastSimProducer","MuonDTHits"), cms.InputTag("fastSimProducer","MuonRPCHits"))
process.simMuonDTDigis.InputCollection = cms.string('fastSimProducerMuonDTHits')
process.simMuonRPCDigis.InputCollection = cms.string('fastSimProducerMuonRPCHits')

process.mixedTripletStepTrackCandidates.simTracks = cms.InputTag("fastSimProducer")
process.trackingParticleNumberOfLayersProducer.simHits = cms.VInputTag("fastSimProducer:TrackerHits")

process.mixSimTracks.input = cms.VInputTag(cms.InputTag("fastSimProducer"))
process.mixSimVertices.input = cms.VInputTag(cms.InputTag("fastSimProducer"))
process.detachedTripletStepTrackCandidates.simTracks = cms.InputTag("fastSimProducer")
process.fastElectronCkfTrackCandidates.simTracks = cms.InputTag("fastSimProducer")

# Not sure if I need all those
process.initialStepTrackCandidates.simTracks = cms.InputTag("fastSimProducer")
process.lowPtTripletStepTrackCandidates.simTracks = cms.InputTag("fastSimProducer")
process.pixelLessStepTrackCandidates.simTracks = cms.InputTag("fastSimProducer")
process.pixelPairStepTrackCandidates.simTracks = cms.InputTag("fastSimProducer")
process.tobTecStepTrackCandidates.simTracks = cms.InputTag("fastSimProducer")
process.AllHcalDigisValidation.simHits = cms.untracked.InputTag("fastSimProducer","HcalHits")
process.AllSimHitsValidation.ModuleLabel = cms.untracked.string('fastSimProducer')
process.mixCollectionValidation.mixObjects.mixSH.input = cms.VInputTag(cms.InputTag("MuonSimHits","MuonCSCHits"), cms.InputTag("MuonSimHits","MuonDTHits"), cms.InputTag("MuonSimHits","MuonRPCHits"), cms.InputTag("fastSimProducer","TrackerHits"))
process.HltVertexValidationVertices.SimVertexCollection = cms.InputTag("fastSimProducer")
process.hltMultiTrackValidator.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.multiTrackValidator.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.tkConversionValidation.simTracks = cms.InputTag("fastSimProducer")
process.trackValidator.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorAllTPEffic.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorAllTPEfficStandalone.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorConversion.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorConversionStandalone.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorConversionTrackingOnly.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorFromPV.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorFromPVAllTP.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorFromPVAllTPStandalone.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorFromPVStandalone.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorGsfTracks.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorLite.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorSeedingTrackingOnly.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorStandalone.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))
process.trackValidatorTrackingOnly.sim = cms.VInputTag(cms.InputTag("fastSimProducer","TrackerHits"))


# load simhit producer
process.load("FastSimulation.FastSimProducer.fastSimProducer_cff")

# Output definition
process.FEVTDEBUGHLTEventContent.outputCommands.extend([
        'keep *_fastSimProducer_*_*',
        'keep FastTrackerRecHits*_*_*_*',
        'keep *_genParticles_*_*',
    ])

#process.FEVTDEBUGHLTEventContent.outputCommands.append(
#        'keep *',
#    )


# Output definition
process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(10485760),
    fileName = cms.untracked.string('muGun_validation.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('dqm_muGun_validation.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)


# Path and EndPath definitions
process.simulation_step = cms.Path(process.fastSimProducer)
process.reconstruction_befmix_step = cms.Path(process.reconstruction_befmix)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.validation_step = cms.EndPath(process.tracksValidationTrackingOnly)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.content=cms.EDAnalyzer('EventContentAnalyzer')
process.content_step = cms.Path(process.content)

# Schedule definition
#process.schedule = cms.Schedule(process.simulation_step,process.FEVTDEBUGHLToutput_step,process.DQMoutput_step)
process.schedule = cms.Schedule(process.simulation_step,process.reconstruction_befmix_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.L1Reco_step,process.reconstruction_step,process.validation_step,process.FEVTDEBUGHLToutput_step,process.DQMoutput_step)

# debugging options
# debug messages will only be printed for packages compiled with following command
# USER_CXXFLAGS="-g -D=EDM_ML_DEBUG" scram b -v # for bash
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger = cms.Service(
#    "MessageLogger",
#    destinations = cms.untracked.vstring('cout'),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG'),
#        ),
#    debugModules = cms.untracked.vstring('fastSimProducer')
#    )
