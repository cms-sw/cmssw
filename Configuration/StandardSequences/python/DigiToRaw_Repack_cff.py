import FWCore.ParameterSet.Config as cms

##
## (1) Remake RAW from ZS tracker digis
##

import EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi
SiStripDigiToZSRaw = EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi.SiStripDigiToRaw.clone(
    InputDigis = cms.InputTag('siStripZeroSuppression', 'VirginRaw'),
    FedReadoutMode = cms.string('ZERO_SUPPRESSED'),
    PacketCode = cms.string('ZERO_SUPPRESSED'),
    CopyBufferHeader = cms.bool(True),
    RawDataTag = cms.InputTag('rawDataCollector')
    )

SiStripDigiToHybridRaw = SiStripDigiToZSRaw.clone(
    PacketCode = cms.string('ZERO_SUPPRESSED10'),
    )

SiStripRawDigiToVirginRaw = SiStripDigiToZSRaw.clone(
    FedReadoutMode = cms.string('VIRGIN_RAW'),
    PacketCode = cms.string('VIRGIN_RAW')
)

##
## (2) Combine new ZS RAW from tracker with existing RAW for other FEDs
##

from EventFilter.RawDataCollector.rawDataCollectorByLabel_cfi import rawDataCollector

rawDataRepacker = rawDataCollector.clone(
    verbose = cms.untracked.int32(0),
    RawCollectionList = cms.VInputTag( cms.InputTag('SiStripDigiToZSRaw'),
                                       cms.InputTag('source'),
                                       cms.InputTag('rawDataCollector'))
    )
hybridRawDataRepacker = rawDataRepacker.clone(
    RawCollectionList = cms.VInputTag( cms.InputTag('SiStripDigiToHybridRaw'),
                                       cms.InputTag('source'),
                                       cms.InputTag('rawDataCollector'))
    )

virginRawDataRepacker = rawDataRepacker.clone(
	RawCollectionList = cms.VInputTag( cms.InputTag('SiStripRawDigiToVirginRaw'))
)

##
## Repacked DigiToRaw Sequence
##

DigiToRawRepack = cms.Sequence( SiStripDigiToZSRaw * rawDataRepacker )
DigiToHybridRawRepack = cms.Sequence( SiStripDigiToHybridRaw * hybridRawDataRepacker )
DigiToVirginRawRepack = cms.Sequence( SiStripRawDigiToVirginRaw * virginRawDataRepacker )
DigiToSplitRawRepack = cms.Sequence( DigiToRawRepack + DigiToVirginRawRepack )
