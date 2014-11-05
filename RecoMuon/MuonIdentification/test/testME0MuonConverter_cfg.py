import FWCore.ParameterSet.Config as cms
#process = cms.Process("ME0SegmentMatching")
process = cms.Process("ME0SegmentMatchingLocalTest")

#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

#process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:startup', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')


#process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023HGCalMuon_cff')
#process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
#process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')



#process.load('Geometry.GEMGeometry.cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi')


## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
#process.load('Configuration.Geometry.GeometryExtended2023HGCalReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023HGCal_cff')
#process.load('Configuration.Geometry.GeometryExtended2023Muon4EtaReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023Muon4Eta_cff')

process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## TrackingComponentsRecord required for matchers
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

## global tag for 2019 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')



process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }
)

process.load('RecoMuon.MuonIdentification.me0MuonReco_cff')
process.p = cms.Path(process.me0MuonReco)

process.PoolSource.fileNames = [
    
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/1EA3C245-00A1-E311-A693-003048FEB9F6.root',
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/3658CD74-F9A0-E311-A114-002590494C22.root',
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/58ABCE86-FDA0-E311-B83D-02163E00E93E.root',
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/8605BDCA-10A1-E311-9FDC-02163E00E805.root',
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/A00B7D25-1CA1-E311-B7FD-02163E00E72D.root'
    #'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Muplus_Pt10-gun_98_dEta0p05_dPhi0p02_secondrun_Dir0p15.root'
    '/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Muplus_Pt10-gun_98_dEta0p05_dPhi0p02.root'
    #'file:13007_SingleMuPt10+SingleMuPt10_Extended2023Muon_GenSimFull+DigiFull_Extended2023Muon+RecoFull_Extended2023Muon+HARVESTFull_Extended2023Muon/step3.root'
]


process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_me0SegmentMatcher_*_*'
        ),
#                              process.AODSIMEventContent,
                              fileName = cms.untracked.string('out_me0_test.root')
)

process.outpath = cms.EndPath(process.o1)
