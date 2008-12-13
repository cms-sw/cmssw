import FWCore.ParameterSet.Config as cms

process = cms.Process("electrons")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("RecoEcal.Configuration.RecoEcal_cff")
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi")
process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")
process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi")
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff")
process.load("RecoEgamma.EgammaElectronProducers.pixelMatchGsfElectronSequence_cff")

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
#       '/store/relval/CMSSW_3_0_0_pre2/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V9_v2/0001/081F51BD-6FB2-DD11-91C4-000423D94700.root',
#       '/store/relval/CMSSW_3_0_0_pre2/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V9_v2/0001/E2426349-1DB4-DD11-B4F7-001617DC1F70.root'
       '/store/relval/CMSSW_3_0_0_pre2/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/822A1F47-1DB4-DD11-A88F-001617C3B6E8.root',
       '/store/relval/CMSSW_3_0_0_pre2/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/C8F991BE-6FB2-DD11-A8ED-000423D98F98.root',
       '/store/relval/CMSSW_3_0_0_pre2/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/E228F0BC-6FB2-DD11-B920-000423D999CA.root',
       '/store/relval/CMSSW_3_0_0_pre2/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/FC810FB2-6FB2-DD11-BC0C-001617DBD5B2.root'
#       '/store/relval/CMSSW_3_0_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/08043F34-71B2-DD11-A8E2-000423D98920.root',
#       '/store/relval/CMSSW_3_0_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/1205FDF4-1CB4-DD11-B31F-000423D8F63C.root',
#       '/store/relval/CMSSW_3_0_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/240B821A-70B2-DD11-980C-0030487A18A4.root',
#       '/store/relval/CMSSW_3_0_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/2496BFC0-6FB2-DD11-B15E-0030487C6062.root',
#       '/store/relval/CMSSW_3_0_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v2/0001/302730DF-70B2-DD11-B716-000423D6A6F4.root'
       )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoSuperClusters*_*_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_*_*_electrons', 
        'keep *HepMCProduct_*_*_*'),
    fileName = cms.untracked.string('electrons.root')
)

process.Timing = cms.Service("Timing")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

process.mylocalreco =  cms.Sequence(process.trackerlocalreco*process.calolocalreco)
process.myglobalreco = cms.Sequence(process.offlineBeamSpot+process.recopixelvertexing*process.ckftracks+process.ecalClusters+process.caloTowersRec*process.vertexreco*process.pixelMatchGsfElectronSequence)
process.p = cms.Path(process.RawToDigi*process.mylocalreco*process.myglobalreco)

process.outpath = cms.EndPath(process.out)
process.GlobalTag.globaltag = 'IDEAL_30X::All'


