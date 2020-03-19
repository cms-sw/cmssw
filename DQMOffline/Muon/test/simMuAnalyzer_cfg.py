import FWCore.ParameterSet.Config as cms

process = cms.Process("Rec")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

#
# DQM
#
process.load("DQMServices.Core.DQM_cfg")

process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cff")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.CommonTopologies.globalTrackingGeometry_cfi")

process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")

# the task
process.load("DQMOffline.Muon.muonAnalyzer_cff")

# the clients
process.load("DQMOffline.Muon.trackResidualsTest_cfi")

process.load("DQMOffline.Muon.muonRecoTest_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/giorgia/6021A7B3-6B15-DD11-8E66-001A92971B94.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50000)
)
process.Timing = cms.Service("Timing")

from DQMServices.Core.DQMQualityTester import DQMQualityTester
process.qTester1 = DQMQualityTester(
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/QualityTests1.xml')
)

process.qTester2 = DQMQualityTester(
    prescaleFactor = cms.untracked.int32(1),
    qtList = cms.untracked.FileInPath('DQMOffline/Muon/data/QualityTests2.xml')
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('muonAnalyzer', 
        'muTrackResidualsTest', 
        'muRecoTest'),
    cout = cms.untracked.PSet(
        muonRecoTest = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        SegmentsTrackAssociator = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        muonAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        seedsAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        segmTrackAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        trackResidualsTest = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        FwkJob = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        threshold = cms.untracked.string('DEBUG'),
        muRecoAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        DEBUG = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        muEnergyDepositAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    categories = cms.untracked.vstring('muonAnalyzer', 
        'seedsAnalyzer', 
        'muEnergyDeposit', 
        'muRecoAnalyzer', 
        'trackResidualsTest', 
        'segmTrackAnalyzer', 
        'SegmentsTrackAssociator', 
        'muonRecoTest'),
    destinations = cms.untracked.vstring('cout')
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('reco-grumm.root')
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True) ## default is false

)
process.p = cms.Path(process.qTester1*process.muonAnalyzer*process.qTester2*process.muTrackResidualsTest*process.muRecoTest)
process.DQM.collectorHost = ''
process.muonAnalyzer.OutputMEsInRootFile = True


