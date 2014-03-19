import FWCore.ParameterSet.Config as cms

process = cms.Process("ME0SegmentMatching")

#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }
)

process.load('RecoMuon.MuonIdentification.me0MuonReco_cff')
process.p = cms.Path(process.me0MuonReco)

process.PoolSource.fileNames = [
    
    '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/1EA3C245-00A1-E311-A693-003048FEB9F6.root',
    '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/3658CD74-F9A0-E311-A114-002590494C22.root',
    '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/58ABCE86-FDA0-E311-B83D-02163E00E93E.root',
    '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/8605BDCA-10A1-E311-9FDC-02163E00E805.root',
    '/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/A00B7D25-1CA1-E311-B7FD-02163E00E72D.root'
    
]


process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
#                              process.AODSIMEventContent,
                              fileName = cms.untracked.string('out_me0_test.root')
)

process.outpath = cms.EndPath(process.o1)
