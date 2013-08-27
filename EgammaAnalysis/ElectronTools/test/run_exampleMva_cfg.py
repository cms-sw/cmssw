import FWCore.ParameterSet.Config as cms

process = cms.Process("EX")
process.load("Configuration.StandardSequences.Services_cff")
#process.load('Configuration.StandardSequences.Geometry_cff')
process.load('Configuration.Geometry.GeometryIdeal_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
##process.GlobalTag.globaltag = 'START42_V12::All'   # CMSSW_4XY
#process.GlobalTag.globaltag = 'START52_V4::All'    # CMSSW_52Y
process.GlobalTag.globaltag = cms.string( 'START53_V7A::All' )

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    #'rfio:/castor/cern.ch/user/b/benedet/Fall11_DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola_0.root'  #CMSSW_4XY
    'root://eoscms//eos/cms//store/relval/CMSSW_5_3_9/RelValZEE/GEN-SIM-RECO/PU_START53_V15A_runMC-v2/00000/40368DF5-2A9D-E211-9817-003048CBA446.root'
    # 'file:/data/benedet/Fall11_DYJetsToLL_TuneZ2_M-50_7TeV-madgraph-tauola_0.root'
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
process.demo = cms.EDAnalyzer("ElectronAnalyzer")

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("electronMVA_AOD.root")
    )

process.pAna = cms.Path(process.demo)

process.schedule = cms.Schedule(process.pAna)





