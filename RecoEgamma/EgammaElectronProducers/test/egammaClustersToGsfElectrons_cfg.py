import FWCore.ParameterSet.Config as cms

process = cms.Process("electrons")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

process.load("RecoEgamma.EgammaElectronProducers.pixelMatchGsfElectronSequence_cff")

process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi")
process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")
process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi")
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff")

process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
#      '/store/relval/CMSSW_2_2_0/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V9_v1/0000/587EC8EF-B4B9-DD11-AC52-001617C3B65A.root',
#      '/store/relval/CMSSW_2_2_0/RelValSingleElectronPt35/GEN-SIM-RECO/IDEAL_V9_v1/0000/9E464300-76B9-DD11-B526-000423D98C20.root'
       '/store/relval/CMSSW_2_2_0/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/92DA70ED-B4B9-DD11-A9BD-001617C3B6FE.root',
       '/store/relval/CMSSW_2_2_0/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/9AF7F8F7-75B9-DD11-B74A-000423D987E0.root',
       '/store/relval/CMSSW_2_2_0/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/9E1383F3-75B9-DD11-B5E6-000423D99996.root',
       '/store/relval/CMSSW_2_2_0/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/F4D16CF0-75B9-DD11-8EFC-000423D98BE8.root'
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep recoSuperClusters*_*_*_*', 
        'keep *_iterativeCone5CaloJets_*_*', 
        'keep *_*_*_electrons', 
        'keep *HepMCProduct_*_*_*'),
    fileName = cms.untracked.string('electrons.root')
)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.EventContent.EventContent_cff")

#process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.ckftracks*process.particleFlowReco*process.gsfElectronAnalysis)
process.p = cms.Path(process.RawToDigi*process.reconstruction*process.pixelMatchGsfElectronSequence)

process.outpath = cms.EndPath(process.out)
process.GlobalTag.globaltag = 'IDEAL_30X::All'


