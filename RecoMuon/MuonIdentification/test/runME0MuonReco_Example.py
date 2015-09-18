import FWCore.ParameterSet.Config as cms
process = cms.Process("ME0SegmentMatchingLocalTest")


## Standard sequence
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

#process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')

process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff')

#process.load('Configuration.Geometry.GeometryExtended2023SHCalNoTaperReco_cff')
#process.load('Configuration.Geometry.GeometryExtended2023SHCalNoTaper_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
#process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

## TrackingComponentsRecord required for matchers
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi')
process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi')

## global tag for 2019 upgrade studies
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2019', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')



# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023SHCal 
#from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023Muon 

#call to customisation function cust_2023SHCal imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
#process = cust_2023SHCal(process)
#process = cust_2023Muon(process)



# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.me0Customs
from SLHCUpgradeSimulations.Configuration.me0Customs import customise 
process = customise(process)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }
)


#process.load('RecoLocalMuon.GEMRecHit.me0RecHits_cfi')
#process.load('RecoLocalMuon.GEMSegments.me0Segments_cfi')
process.load('RecoMuon.MuonIdentification.me0MuonReco_cff')

#process.p = cms.Path(process.me0RecHits*process.me0Segments*process.me0MuonReco)
process.p = cms.Path(process.me0MuonReco)
#process.p = cms.Path(process.me0RecHits*process.me0Segments)

process.PoolSource.fileNames = [
    
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/1EA3C245-00A1-E311-A693-003048FEB9F6.root',
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/3658CD74-F9A0-E311-A114-002590494C22.root',
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/58ABCE86-FDA0-E311-B83D-02163E00E93E.root',
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/8605BDCA-10A1-E311-9FDC-02163E00E805.root',
    #'/store/relval/CMSSW_6_2_0_SLHC8/RelValTTbarLepton_8TeV/GEN-SIM-RECO/DES19_62_V8_UPG2019-v3/00000/A00B7D25-1CA1-E311-B7FD-02163E00E72D.root'
    #'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Muplus_Pt10-gun_98_dEta0p05_dPhi0p02_secondrun_Dir0p15.root'
    #'/store/group/upgrade/muon/ME0GlobalReco/PU_ParallelRun/Muplus_Pt10-gun_98_dEta0p05_dPhi0p02.root'
    #'root://xrootd.unl.edu//store/relval/CMSSW_6_2_0_SLHC22/RelValSingleMuPt10Extended/GEN-SIM-RECO/PU_PH2_1K_FB_V6_SHNoTaperPU140-v1/00000/0204193F-598F-E411-BD90-0025905A48F2.root',
    #'root://xrootd.unl.edu//store/relval/CMSSW_6_2_0_SLHC23_patch1/RelValZMM_14TeV/GEN-SIM-RECO/PH2_1K_FB_V6_UPG2023Muon-v1/00000/0C52947F-D2A7-E411-B999-003048FFD732.root'
    #'root://xrootd.unl.edu//store/relval/CMSSW_6_2_0_SLHC22/RelValSingleMuPt10Extended/GEN-SIM-RECO/PU_PH2_1K_FB_V6_SHNoTaperPU140-v1/00000/4A89321E-618F-E411-9633-0025905A60DA.root'
    #'root://xrootd.unl.edu//store/mc/SHCAL2023Upg14DR/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/GEN-SIM-RECO/PU140bx25_PH2_1K_FB_V4-v1/00000/26982887-3B3E-E411-8532-20CF305B057E.root'
    #'root://xrootd.unl.edu//store/group/upgrade/muon/RecoFolder/DYToMuMu_2023_3Step/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/calabria_DYToMuMu_GEN-SIM-RECO_CMSSW_6_2_0_SLHC23patch1_2023_3Step_OKFS3/2dad437730bcb898314ced9a1ae33ee0/step3_1000_1_BIa.root'
    #'/store/group/upgrade/muon/RecoFolder/DYToMuMu_2023_3Step/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/calabria_DYToMuMu_GEN-SIM-RECO_CMSSW_6_2_0_SLHC23patch1_2023_3Step_OKFS3/2dad437730bcb898314ced9a1ae33ee0/step3_1000_1_BIa.root'
    #'file:/afs/cern.ch/work/d/dnash/ME0Segments/ForRealSegmentsOnly/ReCommit75X/CMSSW_7_5_X_2015-06-29-2300/src/out_digi.root'
    'file:out_local_reco_me0segment.root'
    #'root://xrootd.unl.edu//store/relval/CMSSW_6_2_0_SLHC22/RelValSingleMuPt10Extended/GEN-SIM-RECO/PU_PH2_1K_FB_V6_SHNoTaperPU140-v1/00000/523EEA98-678F-E411-BDBF-002354EF3BE0.root'
    #'file:13007_SingleMuPt10+SingleMuPt10_Extended2023Muon_GenSimFull+DigiFull_Extended2023Muon+RecoFull_Extended2023Muon+HARVESTFull_Extended2023Muon/step3.root'
]


process.o1 = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'keep *',
        'drop *_me0SegmentMatcher_*_*'
        #'drop *',
        ##'keep *_me0SegmentMatching_*_*',
        #'keep *_me0MuonConverting_*_*',
        #'keep *_genParticles_*_*',
        #'keep *_me0Segments_*_*',
        #'keep *_generalTracks_*_*',
        #'keep *_muons_*_*',
        #'keep *_mix_*_*',
        #'keep *_*_MergedTrackTruth_*',
        #'keep *_*_BeamSpot_*',
        #'keep *_offlineBeamSpot_*_*',
        #'keep *_simMuonCSCDigis_*_*',
        #'keep *_simMuonDTDigis_*_*',
        #'keep *_g4SimHits_*_*',
        #'keep *_simSimPixelDigis_*_*',
        #'keep *_simSiPixelDigis_*_*'
        ##'keep *_mix_MergedTrackTruth_*',
        
        ),
#                              process.AODSIMEventContent,
                              fileName = cms.untracked.string('out_me0Reco.root')
                              #fileName = cms.untracked.string('testout.root')
                              )

process.outpath = cms.EndPath(process.o1)
