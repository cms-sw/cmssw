import FWCore.ParameterSet.Config as cms

EcalUnpackerWorkerESProducer = cms.ESProducer("EcalUnpackerWorkerESProducer",
    DCCDataUnpacker = cms.PSet(
        orderedDCCIdList = cms.vint32(1, 2, 3, 4, 5, 
            6, 7, 8, 9, 10, 
            11, 12, 13, 14, 15, 
            16, 17, 18, 19, 20, 
            21, 22, 23, 24, 25, 
            26, 27, 28, 29, 30, 
            31, 32, 33, 34, 35, 
            36, 37, 38, 39, 40, 
            41, 42, 43, 44, 45, 
            46, 47, 48, 49, 50, 
            51, 52, 53, 54),
        tccUnpacking = cms.bool(True),
        srpUnpacking = cms.bool(False),
        syncCheck = cms.bool(False),
        headerUnpacking = cms.bool(False),
        feUnpacking = cms.bool(True),
        orderedFedList = cms.vint32(601, 602, 603, 604, 605, 
            606, 607, 608, 609, 610, 
            611, 612, 613, 614, 615, 
            616, 617, 618, 619, 620, 
            621, 622, 623, 624, 625, 
            626, 627, 628, 629, 630, 
            631, 632, 633, 634, 635, 
            636, 637, 638, 639, 640, 
            641, 642, 643, 644, 645, 
            646, 647, 648, 649, 650, 
            651, 652, 653, 654),
        DCCMapFile = cms.string('EventFilter/EcalRawToDigi/data/DCCMap.txt'),
        feIdCheck = cms.bool(True),
        memUnpacking = cms.bool(False)
    ),
    ElectronicsMapper = cms.PSet(
        numbXtalTSamples = cms.uint32(10),
        numbTriggerTSamples = cms.uint32(1)
    ),
    UncalibRHAlgo = cms.PSet(
         Type = cms.string('EcalUncalibRecHitWorkerWeights')
    ),
    CalibRHAlgo = cms.PSet(
         Type = cms.string('EcalRecHitWorkerSimple'),
         ChannelStatusToBeExcluded = cms.vint32()
    )
)


