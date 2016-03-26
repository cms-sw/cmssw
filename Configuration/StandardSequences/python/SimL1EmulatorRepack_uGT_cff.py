import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

## L1REPACK uGT:  Re-Emulate L1 uGT and repack into RAW


if not (eras.stage2L1Trigger.isChosen()):
    print "L1T WARN:  L1REPACK ALL only supports Stage 2 eras for now."
    print "L1T WARN:  Use a legacy version of L1REPACK for now."
else:
    print "L1T INFO:  L1REPACK CALOuGT will  re-emulate the uGT (Stage-2), and then pack uGT output."

    from EventFilter.L1TRawToDigi.gtStage2Raw_cfi import gtStage2Raw as packGtStage2

    import EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi

    rawDataCollector = EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi.rawDataCollector.clone(
        verbose = cms.untracked.int32(0),
        RawCollectionList = cms.VInputTag(
            cms.InputTag('packGtStage2'),
            cms.InputTag('rawDataCollector', processName=cms.InputTag.skipCurrentProcess()),
            )
        )

    SimL1Emulator = cms.Sequence( SimL1GtEmulatorCore ### only uGT
                                +packGtStage2
                                +rawDataCollector)
