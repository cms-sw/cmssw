import FWCore.ParameterSet.Config as cms

process = cms.Process("JETRECO")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_3_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v3/0002/1AE808B6-5CC0-DE11-AD4A-0030487C6062.root',
        '/store/relval/CMSSW_3_3_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v3/0002/2C34904B-5FC0-DE11-8172-0030487C6090.root',
        '/store/relval/CMSSW_3_3_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v3/0002/389EF6C2-65C0-DE11-8479-003048D3756A.root',
        '/store/relval/CMSSW_3_3_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v3/0002/5296FA7F-5BC0-DE11-9BFE-000423D6BA18.root',
        '/store/relval/CMSSW_3_3_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v3/0002/60496440-61C0-DE11-BF41-000423D9517C.root',
        '/store/relval/CMSSW_3_3_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v3/0002/8432D4DA-5DC0-DE11-81B5-000423D6B48C.root',
        '/store/relval/CMSSW_3_3_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v3/0002/A239A099-62C0-DE11-A063-000423D99A8E.root',
        '/store/relval/CMSSW_3_3_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v3/0002/C46E6E17-60C0-DE11-A79E-000423D991D4.root',
        '/store/relval/CMSSW_3_3_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v3/0003/008104D2-9CC1-DE11-B19F-000423D99A8E.root'
        )
    )


# output
process.load('Configuration/EventContent/EventContent_cff')
process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('testJetReco.root')
)
process.output.outputCommands =  cms.untracked.vstring('drop *')
process.output.outputCommands.append('keep recoCaloJets_*_*_*')
process.output.outputCommands.append('keep recoPFJets_*_*_*')
process.output.outputCommands.append('keep recoGenJets_*_*_*')
process.output.outputCommands.append('keep recoBasicJets_*_*_*')
process.output.outputCommands.append('keep *_genParticles_*_*')
process.output.outputCommands.append('keep *_*_*_JETRECO')


# jet reconstruction
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

process.load('RecoJets.Configuration.GenJetParticles_cff')
process.load('RecoJets.Configuration.RecoGenJets_cff')
process.load('RecoJets.Configuration.RecoJets_cff')
process.load('RecoJets.Configuration.RecoPFJets_cff')
process.load('RecoJets.JetProducers.TracksForJets_cff')
process.load('RecoJets.Configuration.RecoTrackJets_cff')
process.load('RecoJets.Configuration.JetIDProducers_cff')

process.recoJets = cms.Path(process.genParticlesForJets+process.recoGenJets+
                            process.recoJets+
                            process.recoPFJets+
                            process.tracksForJets+process.recoTrackJets+
                            process.recoJetIds
                            )

process.recoAllJets = cms.Path(process.genParticlesForJets+process.recoAllGenJets+
                               process.genParticlesForJetsNoNu+process.recoAllGenJetsNoNu+
                               process.genParticlesForJetsNoMuNoNu+process.recoAllGenJetsNoMuNoNu+
                               process.recoAllJets+
                               process.recoAllPFJets+
                               process.tracksForJets+process.recoAllTrackJets+
                               process.recoAllJetIds
                               )

process.recoAllJetsPUOffsetCorr = cms.Path(process.recoAllJetsPUOffsetCorr
                                           )

process.out = cms.EndPath(process.output)

# schedule
process.schedule = cms.Schedule(process.recoAllJets,process.out)

# Set the threshold for output logging to 'info'
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
#process.MessageLogger.cerr.threshold = cms.untracked.string('DEBUG')
#process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
#process.MessageLogger.debugModules = cms.untracked.vstring('*')
