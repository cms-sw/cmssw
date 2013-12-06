import FWCore.ParameterSet.Config as cms

process = cms.Process("EX")
process.load("Configuration.StandardSequences.Services_cff")
#process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.Geometry.GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup')

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
      '/store/relval/CMSSW_7_0_0_pre8/RelValZEE/GEN-SIM-RECO/PU_START70_V1-v1/00000/10C6FC22-F84A-E311-820E-00304867926C.root',
      '/store/relval/CMSSW_7_0_0_pre8/RelValZEE/GEN-SIM-RECO/PU_START70_V1-v1/00000/4AE7C549-EE4A-E311-9021-00261894396D.root',
      '/store/relval/CMSSW_7_0_0_pre8/RelValZEE/GEN-SIM-RECO/PU_START70_V1-v1/00000/62E749A2-EF4A-E311-8E58-002618943954.root',
      '/store/relval/CMSSW_7_0_0_pre8/RelValZEE/GEN-SIM-RECO/PU_START70_V1-v1/00000/8CA60C13-F14A-E311-AC4A-0026189438F2.root',
      '/store/relval/CMSSW_7_0_0_pre8/RelValZEE/GEN-SIM-RECO/PU_START70_V1-v1/00000/C8C11952-074B-E311-A4F2-002618943836.root'
    ),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
    )


# Event output
process.load("Configuration.EventContent.EventContent_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )

#my analyzer
process.demo = cms.EDAnalyzer("ElectronTestAnalyzer")

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("electronMVA_AOD.root")
    )

process.pAna = cms.Path(process.demo)

process.schedule = cms.Schedule(process.pAna)





