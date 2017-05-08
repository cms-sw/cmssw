import FWCore.ParameterSet.Config as cms

process = cms.Process("JETRECO")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# input
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValTTbarPileUpGENSIMRECO
process.source = cms.Source("PoolSource", fileNames = filesRelValTTbarPileUpGENSIMRECO )
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

# output
process.load('Configuration/EventContent/EventContent_cff')
process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = process.RecoJetsAOD.outputCommands, 
    fileName = cms.untracked.string('testJetRecoRECO.root'),
    dataset = cms.untracked.PSet(
            dataTier = cms.untracked.string(''),
                    filterName = cms.untracked.string('')
                    )
)
process.output.outputCommands.append('drop *_*_*_RECO')
#process.output.outputCommands.append('keep recoCaloJets_*_*_*')
#process.output.outputCommands.append('keep recoPFJets_*_*_*')
#process.output.outputCommands.append('keep recoGenJets_*_*_*')
#process.output.outputCommands.append('keep recoBasicJets_*_*_*')
process.output.outputCommands.append('keep *_*_*_JETRECO')
process.output.outputCommands.append('keep recoPFCandidates_particleFlow_*_*')
#process.output.outputCommands.append('keep *_trackRefsForJets_*_*')
#process.output.outputCommands.append('keep *_generalTracks_*_*')

# jet reconstruction
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.load("RecoJets/Configuration/RecoJetsGlobal_cff")

process.recoJets = cms.Path(process.recoPFJetsWithSubstructure)

process.out = cms.EndPath(process.output)

# schedule
process.schedule = cms.Schedule(process.recoJets,process.out)
