import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalHighEnergyCosmicSkim")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

#process.load("DPGAnalysis.Skims.ecalSkim_cfi")
process.load("DPGAnalysis/Skims/ecalSkim_cfi")

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(0),
    debugFlag = cms.untracked.bool(False),
    secondaryFileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RAW/MW30_v3/000/052/496/408FF22A-C258-DD11-9B7A-000423D992A4.root'),
    fileNames = cms.untracked.vstring('/store/data/Commissioning08/Cosmics/RECO/CRUZET3_V2P_MW30_v1/0000/3CF69559-DD58-DD11-8155-001617DF785A.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET3_V2P_MW30_v1/0000/6C8B2659-DD58-DD11-A992-000423D9890C.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET3_V2P_MW30_v1/0000/7076DAE9-DB58-DD11-BEC7-000423D9863C.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET3_V2P_MW30_v1/0000/96B813FD-DB58-DD11-A231-001617E30F50.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET3_V2P_MW30_v1/0000/9A2D7AF0-DB58-DD11-9DF5-001617DF785A.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET3_V2P_MW30_v1/0000/B28F55C2-DC58-DD11-82A2-000423D98B5C.root', 
        '/store/data/Commissioning08/Cosmics/RECO/CRUZET3_V2P_MW30_v1/0000/B411F7C1-DF58-DD11-A06F-001617C3B6DC.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.out = cms.OutputModule("PoolOutputModule",
    filterName = cms.untracked.string('skimming'),
    fileName = cms.untracked.string('testSkim.root'),
    outputCommands = cms.untracked.vstring('keep *'),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)

process.p = cms.Path(process.skimming)
process.e = cms.EndPath(process.out)

