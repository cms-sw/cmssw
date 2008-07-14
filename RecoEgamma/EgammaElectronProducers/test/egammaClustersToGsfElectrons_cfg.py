import FWCore.ParameterSet.Config as cms

process = cms.Process("electrons")

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
    fileNames = cms.untracked.vstring('/store/relval/2008/7/13/RelVal-RelValSingleElectronPt35-1215820444-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-1215820444-IDEAL_V5-2nd-IDEAL_V5-unmerged/0000/14AD4148-F850-DD11-A295-000423D98DD4.root', 
        '/store/relval/2008/7/13/RelVal-RelValSingleElectronPt35-1215820444-IDEAL_V5-2nd/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre8-RelVal-1215820444-IDEAL_V5-2nd-IDEAL_V5-unmerged/0000/F082A8E5-F650-DD11-B6DA-0019DB2F3F9B.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_*_*_electrons'),
    fileName = cms.untracked.string('electrons.root')
)

process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.newSeedFromPairs*process.newSeedFromTriplets*process.newCombinedSeeds*process.pixelMatchGsfElectronSequence)
process.outpath = cms.EndPath(process.out)
process.GlobalTag.globaltag = 'IDEAL_V5::All'


