import FWCore.ParameterSet.Config as cms

process = cms.Process("readseeds")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff")
process.load("RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff")
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi")
process.load("RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi")
process.load("RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi")
process.load("RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi")
process.load("RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi")
process.load("RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff")
process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source ("PoolSource",
    fileNames = cms.untracked.vstring (
#   '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/00E17DCF-8D82-DD11-BF03-000423D987E0.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/16427204-8E82-DD11-90F0-000423D985E4.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/2A800506-8E82-DD11-BE3D-001617C3B79A.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/329606EC-8D82-DD11-9B9F-001617C3B77C.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/44FC807B-8E82-DD11-BF6D-000423DD2F34.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/58853AB7-8D82-DD11-BC45-001617E30D40.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/68152B0D-8E82-DD11-ACDD-000423D986A8.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/861DDBD2-8D82-DD11-AF6A-001617DBD556.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/9E0929A9-8D82-DD11-BB3D-000423D9997E.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/AE71B0F9-8D82-DD11-9D05-001617E30E28.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/B24C31D6-8D82-DD11-906E-001617C3B6DE.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/B619FFE6-8D82-DD11-84AE-001617C3B710.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/B8775C9D-8D82-DD11-BBBF-00161757BF42.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/BC0F2390-8D82-DD11-86D3-000423D60FF6.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/CA8B36D4-8D82-DD11-AB7E-000423D986A8.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/F237E3AD-8D82-DD11-9C74-000423D986A8.root',       '/store/relval/CMSSW_2_1_8/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0003/1643F3F0-A682-DD11-B9CE-001617C3B778.root'
    )
)
process.electronSeedAnalysis = cms.EDAnalyzer("ElectronSeedAnalyzer",
    inputCollection = cms.InputTag("ecalDrivenElectronSeeds"),
)

process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.newSeedFromPairs*process.newSeedFromTriplets*process.newCombinedSeeds*process.ecalDrivenElectronSeeds*process.electronSeedAnalysis)
process.GlobalTag.globaltag = 'STARTUP_V7::All'
#process.GlobalTag.globaltag = 'IDEAL_V7::All'


