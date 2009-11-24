import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP31X_V8::All')

process.source = cms.Source (
    "PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/D8D6F277-C5C7-DE11-A59F-002618943962.root',
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/C8072F59-59C8-DE11-BB0A-00261894393B.root',
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/BC410BBC-C4C7-DE11-BA4F-002618FDA237.root',
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/32E84E16-C4C7-DE11-A181-002618943833.root',
    '/store/relval/CMSSW_3_3_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V8-v2/0000/0295FB14-C4C7-DE11-834B-002618943862.root',
    )
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

# IC5, SC5 and AK5
process.p = cms.Path (
    process.ZSPJetCorrectionsIcone5 *
    process.JetPlusTrackCorrectionsIcone5 *
    process.ZSPJetCorrectionsSisCone5 *
    process.JetPlusTrackCorrectionsSisCone5 *
    process.ZSPJetCorrectionsAntiKt5 *
    process.JetPlusTrackCorrectionsAntiKt5 
    )

process.o = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(
    'keep *',
    )
    )
process.e = cms.EndPath( process.o )
