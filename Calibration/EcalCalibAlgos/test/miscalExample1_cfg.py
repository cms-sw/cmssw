#Example on how to miscalibrate rec hits starting from uncalibrated rechits.

import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.load("Configuration.StandardSequences.Services_cff")

process.load("Configuration.StandardSequences.GeometryPilot2_cff")

process.load("Configuration.StandardSequences.MagneticField_38T_cff")

process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_noesprefer_cff")
process.GlobalTag.globaltag = 'IDEAL_V9::All'

#Assume root file contains EcalRecHits
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_2_1_6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/0628F203-C078-DD11-8B63-001D09F25208.root',
'/store/relval/CMSSW_2_1_6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/44C01E47-C078-DD11-A3C1-001D09F24EC0.root',
'/store/relval/CMSSW_2_1_6/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V9_v1/0000/EA6E5AFD-BF78-DD11-93F9-001D09F2983F.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)

process.CaloMiscalibTools = cms.ESSource("CaloMiscalibTools",
    fileNameEndcap = cms.untracked.string('miscalib_endcap_0.05.xml'),
    fileNameBarrel = cms.untracked.string('miscalib_barrel_0.05.xml')
)
process.prefer("CaloMiscalibTools")

process.testMiscalibration = cms.EDAnalyzer("miscalibExample",
    rootfile = cms.untracked.string('miscalibExample1.root'),
    correctedHybridSuperClusterProducer = cms.string('correctedHybridSuperClusters'),
    correctedHybridSuperClusterCollection = cms.string('')
)

process.p = cms.Path(process.RawToDigi*process.calolocalreco*process.ecalClusters*process.testMiscalibration)
