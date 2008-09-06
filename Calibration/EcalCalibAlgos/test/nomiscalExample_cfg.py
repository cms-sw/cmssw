#Example on how to miscalibrate rec hits starting from uncalibrated rechits.

import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.GeometryPilot2_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.GlobalTag.globaltag = 'IDEAL_V9::All'

#Assume root file contains EcalRecHits
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_2_1_6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/048E4718-C078-DD11-9CD1-001D09F29533.root',
'/store/relval/CMSSW_2_1_6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/60D85937-C078-DD11-93AA-001D09F28755.root',
'/store/relval/CMSSW_2_1_6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/7E294A5A-C078-DD11-8BFB-001D09F24493.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)

#process.CaloMiscalibTools = cms.ESSource("CaloMiscalibTools",
#    fileNameEndcap = cms.untracked.string('miscalib_endcap_0.05.xml'),
#    fileNameBarrel = cms.untracked.string('miscalib_barrel_0.05.xml')
#)
#process.prefer("CaloMiscalibTools")

process.testMiscalibration = cms.EDAnalyzer("miscalibExample",
    rootfile = cms.untracked.string('nomiscalibExample.root'),
    correctedHybridSuperClusterProducer = cms.string('correctedHybridSuperClusters'),
    correctedHybridSuperClusterCollection = cms.string('')
)

process.p = cms.Path(process.ecalClusters*process.testMiscalibration)
