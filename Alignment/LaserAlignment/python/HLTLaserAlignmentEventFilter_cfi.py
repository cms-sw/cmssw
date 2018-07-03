import FWCore.ParameterSet.Config as cms

LaserAlignmentEventFilter = cms.EDFilter("LaserAlignmentEventFilter",
  FedInputTag = cms.InputTag( 'source' )
)

LaserAlignmentEventFilter.FED_IDs = cms.vint32()

LaserAlignmentEventFilter.DET_IDs = cms.vint32()

LaserAlignmentEventFilter.SIGNAL_IDs = cms.vint32()

LaserAlignmentEventFilter.SINGLE_CHANNEL_THRESH = cms.uint32(11);
LaserAlignmentEventFilter.CHANNEL_COUNT_THRESH = cms.uint32(8);

