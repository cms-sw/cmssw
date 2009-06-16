import FWCore.ParameterSet.Config as cms

process = cms.Process("RECO")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.L1Emulator_cff")
process.load("Configuration.StandardSequences.L1TriggerDefaultMenu_cff")
process.load("Configuration.StandardSequences.DigiToRaw_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

#global tags for conditions data: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#31X_pre_releases_and_integration
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_31X::All'

# For including Standard Reco + Heavy Ion Objects
process.load("RecoHI.Configuration.Reconstruction_HI_cff")
#process.load("HeavyIonsAnalysis.Configuration.HIAnalysisEventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/pgun_jpsi2muons_d20080423/pgun_jpsi2muons_d20080423_r000001.root')
fileNames = cms.untracked.vstring(           "rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/hydjet_sim_x2_c5_d20080425/hydjet_sim_x2_c5_d20080425_r000002.root")
#fileNames = cms.untracked.vstring("dcache:/pnfs/cmsaf.mit.edu/hibat/cms/users/davidlw/HYDJET_Minbias_4TeV_31X/sim/HYDJET_Minbias_4TeV_seq11_31X.root")
)

process.load("Configuration.StandardSequences.MixingNoPileUp_cff")

#from HeavyIonsAnalysis.Configuration.EventEmbedding_cff import *
#process.mix=mixSim
#process.mix.input.fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/cms/store/cmshi/mc/sim/pgun_jpsi2muons_d20080423/pgun_jpsi2muons_d20080423_r000005.root')

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        simMuonRPCDigis = cms.untracked.uint32(6),
        simEcalUnsuppressedDigis = cms.untracked.uint32(8),
        simSiStripDigis = cms.untracked.uint32(7),
        mix = cms.untracked.uint32(4),
        simHcalUnsuppressedDigis = cms.untracked.uint32(9),
        simMuonCSCDigis = cms.untracked.uint32(6),
        VtxSmeared = cms.untracked.uint32(2),
        g4SimHits = cms.untracked.uint32(3),
        simMuonDTDigis = cms.untracked.uint32(6),
        simSiPixelDigis = cms.untracked.uint32(7)
    ),
    sourceSeed = cms.untracked.uint32(1)
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    oncePerEventMode = cms.untracked.bool(False),
    ignoreTotal = cms.untracked.int32(0)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('towerMaker', 
        'caloTowers', 
        'iterativeConePu5CaloJets'),
    destinations = cms.untracked.vstring('cout', 
        'cerr'),
    cerr = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    fwkJobReports = cms.untracked.vstring('FrameworkJobReport.xml')
)

process.Timing = cms.Service("Timing")

process.output = cms.OutputModule("PoolOutputModule",
#    process.HIRecoObjects,
    compressionLevel = cms.untracked.int32(2),
    commitInterval = cms.untracked.uint32(1),
    fileName = cms.untracked.string('jpsi2muons_PbPb_RECO.root')
)

process.p = cms.Path(process.mix*process.doAllDigi*process.L1Emulator*process.DigiToRaw*process.RawToDigi*process.reconstruct_PbPb)
process.outpath = cms.EndPath(process.output)
#process.output.outputCommands.append('keep *_*_*_RECO')


