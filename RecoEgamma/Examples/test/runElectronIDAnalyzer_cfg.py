import FWCore.ParameterSet.Config as cms

process = cms.Process("runElectronID")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(1),
    debugFlag = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_2/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V6_v1/0000/16A74923-9368-DD11-AB9B-000423D6CA72.root')
)

process.electronIdAnalyzer = cms.EDAnalyzer("ElectronIDAnalyzer",
    electronProducer = cms.string('gsfElectrons'),
    electronLabelRobustTight = cms.string('eidRobustTight'),
    electronLabelTight = cms.string('eidTight'),
    electronLabelLoose = cms.string('eidLoose'),
    electronLabelRobustLoose = cms.string('eidRobustLoose')
)

process.p = cms.Path(process.electronIdAnalyzer)


