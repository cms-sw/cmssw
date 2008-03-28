import FWCore.ParameterSet.Config as cms

looper = cms.Looper("ZeeCalibration",
    electronCollection = cms.string(''),
    #
    #     Specific ZIterativeAlgorithm parameters       #    #    #    #    #    #    #    #    #    #    #    #    #
    ZCalib_InvMass = cms.untracked.string('SCMass'),
    outputFile = cms.string('myHistograms_Spring07.root'),
    erechitCollection = cms.string('EcalRecHitsEE'),
    scIslandCollection = cms.string(''),
    scIslandProducer = cms.string('electronRecalibSCAssociator'),
    mcProducer = cms.untracked.string('zeeFilter'),
    electronSelection = cms.untracked.uint32(0),
    scProducer = cms.string('correctedHybridSuperClusters'),
    ZCalib_CalibType = cms.untracked.string('RING'),
    initialMiscalibrationBarrel = cms.untracked.string(''),
    maxLoops = cms.untracked.uint32(10),
    electronProducer = cms.string('electronRecalibSCAssociator'),
    erechitProducer = cms.string('recalibRechit'),
    wantEtaCorrection = cms.untracked.bool(True),
    initialMiscalibrationEndcap = cms.untracked.string(''),
    ZCalib_nCrystalCut = cms.untracked.int32(-1),
    rechitProducer = cms.string('recalibRechit'),
    scCollection = cms.string('recalibSC'),
    rechitCollection = cms.string('EcalRecHitsEB')
)


