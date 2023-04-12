import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
process = cms.Process("testSkimming", Run3)

# Message logger service

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.enable = False
process.MessageLogger.DisappearingMuonsSkimming=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),                       
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(100)
                                   ),                                                      
    DMRChecker = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    DisappearingMuonsSkimming = cms.untracked.PSet( limit = cms.untracked.int32(-1))
    )

readFiles = cms.untracked.vstring(
    '/store/relval/CMSSW_13_0_0_pre3/RelValZMM_14/GEN-SIM-RECO/PU_130X_mcRun3_2022_realistic_v2-v1/00000/04f597b5-14d3-44c8-851e-c5f46657d92a.root',
)   

process.source = cms.Source("PoolSource",fileNames = readFiles)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32((800)))

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '130X_mcRun3_2022_realistic_v2', '')

process.load("Configuration.Skimming.PDWG_EXODisappMuon_cff")
# Additional output definition
process.SKIMStreamEXODisappMuon = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('EXODisappMuonPath')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('AODSIM'),
        filterName = cms.untracked.string('EXODisappMuon')
        #filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('EXODisappMuon.root'),
    outputCommands = process.EXODisappMuonSkimContent.outputCommands
)

process.EXODisappMuonPath = cms.Path(process.EXODisappMuonSkimSequence)
process.SKIMStreamEXODisappMuonOutPath = cms.EndPath(process.SKIMStreamEXODisappMuon)

# Schedule definition
process.schedule = cms.Schedule(process.EXODisappMuonPath,process.SKIMStreamEXODisappMuonOutPath)
