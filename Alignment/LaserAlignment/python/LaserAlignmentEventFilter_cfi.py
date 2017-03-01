import FWCore.ParameterSet.Config as cms


from Alignment.LaserAlignment.LaserAlignmentDefs_cff import *

LaserAlignmentEventFilter = cms.EDFilter("LaserAlignmentEventFilter",
  FedInputTag = cms.InputTag( 'source' )
)

LaserAlignmentEventFilter.FED_IDs = cms.vint32()
LaserAlignmentEventFilter.FED_IDs.extend(FED_TECp)
LaserAlignmentEventFilter.FED_IDs.extend(FED_TECm)
LaserAlignmentEventFilter.FED_IDs.extend(FED_AT_TOB)
LaserAlignmentEventFilter.FED_IDs.extend(FED_AT_TIB)
LaserAlignmentEventFilter.FED_IDs.extend(FED_AT_TECp)
LaserAlignmentEventFilter.FED_IDs.extend(FED_AT_TECm)

LaserAlignmentEventFilter.DET_IDs = cms.vint32()
LaserAlignmentEventFilter.DET_IDs.extend(DET_TECp)
LaserAlignmentEventFilter.DET_IDs.extend(DET_TECm)
LaserAlignmentEventFilter.DET_IDs.extend(DET_AT_TOB)
LaserAlignmentEventFilter.DET_IDs.extend(DET_AT_TIB)
LaserAlignmentEventFilter.DET_IDs.extend(DET_AT_TECp)
LaserAlignmentEventFilter.DET_IDs.extend(DET_AT_TECm)

LaserAlignmentEventFilter.SIGNAL_IDs = cms.vint32()
LaserAlignmentEventFilter.SIGNAL_IDs.extend(SIGNAL_IDs_TECp_R4)
LaserAlignmentEventFilter.SIGNAL_IDs.extend(SIGNAL_IDs_TECp_R6)
LaserAlignmentEventFilter.SIGNAL_IDs.extend(SIGNAL_IDs_TECm_R4)
LaserAlignmentEventFilter.SIGNAL_IDs.extend(SIGNAL_IDs_TECm_R6)
LaserAlignmentEventFilter.SIGNAL_IDs.extend(SIGNAL_IDs_AT_TOB)
LaserAlignmentEventFilter.SIGNAL_IDs.extend(SIGNAL_IDs_AT_TIB)
LaserAlignmentEventFilter.SIGNAL_IDs.extend(SIGNAL_IDs_AT_TECp)
LaserAlignmentEventFilter.SIGNAL_IDs.extend(SIGNAL_IDs_AT_TECm)

LaserAlignmentEventFilter.SINGLE_CHANNEL_THRESH = cms.uint32(11);
LaserAlignmentEventFilter.CHANNEL_COUNT_THRESH = cms.uint32(8);

