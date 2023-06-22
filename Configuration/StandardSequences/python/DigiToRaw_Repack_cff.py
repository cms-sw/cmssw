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

DigiToRawRepackTask = cms.Task(SiStripDigiToZSRaw, rawDataRepacker)
DigiToHybridRawRepackTask = cms.Task(SiStripDigiToHybridRaw, hybridRawDataRepacker)
DigiToVirginRawRepackTask = cms.Task(SiStripRawDigiToVirginRaw, virginRawDataRepacker)

DigiToRawRepack = cms.Sequence( DigiToRawRepackTask )
DigiToHybridRawRepack = cms.Sequence( DigiToHybridRawRepackTask )
DigiToVirginRawRepack = cms.Sequence( DigiToVirginRawRepackTask )
DigiToSplitRawRepack = cms.Sequence( DigiToRawRepackTask, DigiToVirginRawRepackTask )

from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import siStripDigis
siStripDigisHLT = siStripDigis.clone(ProductLabel = "rawDataRepacker")

from RecoLocalTracker.Configuration.RecoLocalTracker_cff import siStripZeroSuppressionHLT

from RecoLocalTracker.SiStripClusterizer.DefaultClusterizer_cff import *
siStripClustersHLT = cms.EDProducer("SiStripClusterizer",
                                    Clusterizer = DefaultClusterizer,
                                    DigiProducersList = cms.VInputTag(
                                        cms.InputTag('siStripDigisHLT','ZeroSuppressed'),
                                        cms.InputTag('siStripZeroSuppressionHLT','VirginRaw'),
                                        cms.InputTag('siStripZeroSuppressionHLT','ProcessedRaw'),
                                        cms.InputTag('siStripZeroSuppressionHLT','ScopeMode')),
                                )

from RecoLocalTracker.SiStripClusterizer.SiStripClusters2ApproxClusters_cff import * 

from EventFilter.Utilities.EvFFEDExcluder_cfi import EvFFEDExcluder as _EvFFEDExcluder
rawPrimeDataRepacker = _EvFFEDExcluder.clone(
    src = 'rawDataCollector',
    fedsToExclude = [foo for foo in range(50, 490)]
)

hltScalersRawToDigi =  cms.EDProducer( "ScalersRawToDigi",
   scalersInputTag = cms.InputTag( "rawDataRepacker" )
)

DigiToApproxClusterRawTask = cms.Task(siStripDigisHLT,siStripZeroSuppressionHLT,hltScalersRawToDigi,hltBeamSpotProducer,siStripClustersHLT,hltSiStripClusters2ApproxClusters,rawPrimeDataRepacker)
DigiToApproxClusterRaw = cms.Sequence(DigiToApproxClusterRawTask)
