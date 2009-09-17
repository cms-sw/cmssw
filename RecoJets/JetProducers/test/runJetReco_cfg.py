import FWCore.ParameterSet.Config as cms

process = cms.Process("JETRECO")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10))
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_3_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0015/222A169F-2E9F-DE11-83BA-0030487C6090.root',
        '/store/relval/CMSSW_3_3_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0015/6E4FCE3C-449F-DE11-9B71-001D09F26509.root',
        '/store/relval/CMSSW_3_3_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0015/B419F819-509F-DE11-82DD-003048D3756A.root',
        '/store/relval/CMSSW_3_3_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0015/B8856387-7A9F-DE11-B764-001D09F24353.root',
        '/store/relval/CMSSW_3_3_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0015/E6D38CA0-349F-DE11-8E54-001D09F25217.root',
        '/store/relval/CMSSW_3_3_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_31X_V8-v1/0015/EC3610D1-4E9F-DE11-86B4-0019B9F72CE5.root'
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
