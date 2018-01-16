import FWCore.ParameterSet.Config as cms

PedsMon = DQMStep1Module('SiStripMonitorPedestals',
    OutputMEsInRootFile = cms.bool(False),
    StripQualityLabel = cms.string(''),
    RunTypeFlag = cms.string('CalculatedPlotsOnly'), ##Options : ConDBPlotsOnly , CalculatedPlotsOnly, AllPlots

    DigiProducer = cms.string('siStripDigis'),
    PedestalsPSet = cms.PSet(
        MaskDeadCut = cms.double(0.7),
        MaskCalculationFlag = cms.int32(1),
        NumberOfEventsForInit = cms.int32(200),
        MaskNoiseCut = cms.double(6.0),
        useDB = cms.bool(False),
        NumCMstripsInGroup = cms.int32(128),
        CutToAvoidSignal = cms.double(3.0),
        NumberOfEventsForIteration = cms.int32(100),
        CalculatorAlgorithm = cms.string('TT6'),
        CMType = cms.string('TT6'),
        MaskTruncationCut = cms.double(0.05)
    ),
    UseFedKey = cms.untracked.bool(True),
    OutPutFileName = cms.string('SiStripPedestal.root')
)


