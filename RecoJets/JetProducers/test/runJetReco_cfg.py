import FWCore.ParameterSet.Config as cms

process = cms.Process("JETRECO")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# input
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))
process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0008/FA9E955C-8857-DE11-85DC-001D09F28E80.root',
    '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0008/F09CF4A5-0458-DE11-B9F2-001D09F23C73.root',
    '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0008/F0217F6A-8657-DE11-9B26-001D09F2910A.root',
    '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0008/CA3AC3A0-8757-DE11-A848-001D09F25325.root',
    '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0008/C2B2685B-8557-DE11-80E8-000423D6C8E6.root',
    '/store/relval/CMSSW_3_1_0_pre10/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/IDEAL_31X_v1/0008/1637B9F2-8857-DE11-A04E-001D09F2538E.root')
    )


# output
process.load('Configuration/EventContent/EventContent_cff')
process.output = cms.OutputModule(
    "PoolOutputModule",
    outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('/uscms_data/d2/rappocc/JetMET/reco/FastJetReco_phase6.root')
)
process.output.outputCommands.append('keep *_*_*_JETRECO');


# jet reconstruction
process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

process.load('RecoJets.Configuration.GenJetParticles_cff')
process.load('RecoJets.Configuration.RecoGenJets_cff')
process.load('RecoJets.Configuration.RecoJets_cff')
process.load('RecoJets.Configuration.RecoPFJets_cff')
process.load('RecoJets.JetProducers.TracksForJets_cff')
process.load('RecoJets.Configuration.RecoTrackJets_cff')

process.recoJets = cms.Path(process.genParticlesForJets+process.recoGenJets+
                            process.recoJets+
                            process.recoPFJets+
                            process.tracksForJets+process.recoTrackJets
                            )

process.recoAllJets = cms.Path(process.genParticlesForJets+process.recoAllGenJets+
                               process.recoAllJets+
                               process.recoAllPFJets+
                               process.tracksForJets+process.recoAllTrackJets
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
