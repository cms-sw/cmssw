import FWCore.ParameterSet.Config as cms

looper = cms.Looper("ZeeCalibration",
    electronCollection = cms.string(''),
    ZCalib_InvMass = cms.untracked.string('SCMass'),
    scIslandCollection = cms.string(''),
    erechitCollection = cms.string('EcalRecHitsEE'),
    initialMiscalibrationBarrel = cms.untracked.string(''),
    calibMode = cms.string('RING'),
    initialMiscalibrationEndcap = cms.untracked.string(''),
    HLTriggerResults = cms.InputTag("TriggerResults","","HLT"),
    rechitCollection = cms.string('EcalRecHitsEB'),
    ZCalib_CalibType = cms.untracked.string('RING'),
    ZCalib_nCrystalCut = cms.untracked.int32(-1),
    maxLoops = cms.untracked.uint32(10),
    erechitProducer = cms.string('recalibRechit'),
    wantEtaCorrection = cms.untracked.bool(True),
    outputFile = cms.string('myHistograms_Spring07.root'),
    electronSelection = cms.untracked.uint32(0),
    scProducer = cms.string('correctedHybridSuperClusters'),
    rechitProducer = cms.string('recalibRechit'),
    scIslandProducer = cms.string('electronRecalibSCAssociator'),
    mcProducer = cms.untracked.string('zeeFilter'),
    electronProducer = cms.string('electronRecalibSCAssociator'),
    scCollection = cms.string('recalibSC')
)



