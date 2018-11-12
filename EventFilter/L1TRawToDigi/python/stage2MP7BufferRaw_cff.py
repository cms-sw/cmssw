import FWCore.ParameterSet.Config as cms

from EventFilter.L1TRawToDigi.stage2MP7BufferRaw_cfi import *
from EventFilter.L1TRawToDigi.stage2DemuxMP7BufferRaw_cfi import *
from EventFilter.L1TRawToDigi.stage2GTMP7BufferRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *

rawDataCollector.RawCollectionList = cms.VInputTag(
    cms.InputTag('stage2MPRaw'),
    cms.InputTag('stage2DemuxRaw'),
    cms.InputTag('stage2GTRaw')
)

stage2MP7BufferRaw = cms.Sequence(
    stage2MPRaw
    +stage2DemuxRaw
    +stage2GTRaw
    +rawDataCollector
)
