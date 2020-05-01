import FWCore.ParameterSet.Config as cms

process = cms.Process("TICLDEBUG")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:step3.root'
    )
)

process.load("RecoHGCal.TICL.ticlDebugger_cfi")
process.load("SimGeneral.Debugging.caloParticleDebugger_cfi")

process.p = cms.Path(process.ticlDebugger+process.caloParticleDebugger)

