import FWCore.ParameterSet.Config as cms

process = cms.Process("electrons")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
process.load("Configuration.StandardSequences.VtxSmearedGauss_cff") 
process.load("Configuration.StandardSequences.Generator_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff") 
process.load("Configuration.StandardSequences.L1Emulator_cff") 
process.load("Configuration.StandardSequences.Digi_cff")
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("EmptySource")

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(35.),
        MinPt = cms.double(35.),
        PartID = cms.vint32(11),
        MaxEta = cms.double(2.5),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(-2.5),
        MinPhi = cms.double(-3.14159265359) ## in radians

    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    psethack = cms.string('single electron pt 35'),
    AddAntiParticle = cms.bool(True),
    firstRun = cms.untracked.uint32(1)
)

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
  sourceSeed = cms.untracked.uint32(414000),
  moduleSeeds = cms.PSet(
   generator = cms.untracked.uint32(1041963),   
   VtxSmeared = cms.untracked.uint32(414001),
   g4SimHits = cms.untracked.uint32(414002),
   mix = cms.untracked.uint32(414003),
   simEcalUnsuppressedDigis = cms.untracked.uint32(414004),
   simMuonCSCDigis = cms.untracked.uint32(414005),
   simSiPixelDigis = cms.untracked.uint32(414006),
   simHcalUnsuppressedDigis = cms.untracked.uint32(414007),
   simMuonDTDigis = cms.untracked.uint32(414008),
   simSiStripDigis = cms.untracked.uint32(414009),
   simMuonRPCDigis = cms.untracked.uint32(414010)
  )
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *', 
#        'keep recoSuperClusters*_*_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep reco*_*_*_electrons', 
        'keep *HepMCProduct_*_*_*'),
    fileName = cms.untracked.string('SingleElectronPt35.root')
)

process.Timing = cms.Service("Timing")

process.simulation = cms.Sequence(process.psim*process.pdigi*process.L1Emulator*process.DigiToRaw)
process.mylocalreco =  cms.Sequence(process.trackerlocalreco*process.calolocalreco)
process.myglobalreco = cms.Sequence(process.offlineBeamSpot+process.recopixelvertexing*process.ckftracks+process.ecalClusters+process.caloTowersRec*process.vertexreco*process.particleFlowCluster)
process.myelectronseeding = cms.Sequence(process.trackerDrivenElectronSeeds*process.ecalDrivenElectronSeeds*process.electronMergedSeeds)
process.myelectrontracking = cms.Sequence(process.electronCkfTrackCandidates*process.electronGsfTracks)
process.p = cms.Path(process.generator*process.VertexSmearing*process.simulation*process.RawToDigi*process.mylocalreco*process.myglobalreco*process.myelectronseeding*process.myelectrontracking*process.particleFlowReco*process.pixelMatchGsfElectrons*process.gsfElectronAnalysis)

# to switch on only one seeding mode
#process.electronCkfTrackCandidates.src = cms.InputTag('ecalDrivenElectronSeeds')
#process.electronCkfTrackCandidates.src = cms.InputTag('trackerDrivenElectronSeeds:SeedsForGsf')

process.outpath = cms.EndPath(process.out)
process.GlobalTag.globaltag = 'IDEAL_30X::All'

#to simulate simple gaussian beam spot with no offset
#process.load("RecoVertex.BeamSpotProducer.BeamSpotFakeConditionsSimpleGaussian_cff")
#process.es_prefer = cms.ESPrefer("BeamSpotFakeConditions")
