import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP31X_V4::All')

process.source = cms.Source (
    "PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/DCAE40E8-CA78-DE11-8F20-001D09F2305C.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/D099AB85-CA78-DE11-9A5E-001D09F2503C.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/B2E190DF-CA78-DE11-9F35-001D09F2532F.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/90A07B7B-CA78-DE11-A4B7-001D09F26C5C.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/4C5F86D1-CB78-DE11-8650-000423D6B42C.root',
    '/store/relval/CMSSW_3_1_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP31X_V2-v1/0007/2247873A-B378-DE11-8F5B-001D09F24664.root',
    )
    )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.load("JetMETCorrections.Configuration.ZSPJetCorrections219_cff")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")

# IC5 only (no SC5 or AK5)
process.p = cms.Path (
    process.ZSPJetCorrections *
    process.JetPlusTrackCorrections 
    )

process.o = cms.OutputModule(
    "PoolOutputModule",
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(
    'keep *',
    )
    )
process.e = cms.EndPath( process.o )
