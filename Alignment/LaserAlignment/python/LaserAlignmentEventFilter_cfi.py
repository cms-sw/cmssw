import FWCore.ParameterSet.Config as cms

import Alignment.LaserAlignment.LaserAlignmentDefs_cff as LasDefs

LaserAlignmentEventFilter = cms.EDFilter("LaserAlignmentEventFilter",
FedInputTag = cms.string( 'source' ),
      SIGNAL_Filter = cms.bool(True),
      SINGLE_CHANNEL_THRESH = cms.uint32(11),
      CHANNEL_COUNT_THRESH = cms.uint32(8),
      FED_IDs = LasDefs.FED_ALL,
      SIGNAL_IDs = LasDefs.SIGNAL_IDs_ALL
)

#LaserAlignmentEventFilter.FED_IDs.extend(LasDefs.FED_TECp)
#LaserAlignmentEventFilter.FED_IDs.extend(LasDefs.FED_TECm)
#LaserAlignmentEventFilter.FED_IDs.extend(LasDefs.FED_AT_TOB)
#LaserAlignmentEventFilter.FED_IDs.extend(LasDefs.FED_AT_TIB)
#LaserAlignmentEventFilter.FED_IDs.extend(LasDefs.FED_AT_TECp)
#LaserAlignmentEventFilter.FED_IDs.extend(LasDefs.FED_AT_TECm)

#LaserAlignmentEventFilter.DET_IDs.extend(LasDefs.DET_TECp)
#LaserAlignmentEventFilter.DET_IDs.extend(LasDefs.DET_TECm)
#LaserAlignmentEventFilter.DET_IDs.extend(LasDefs.DET_AT_TOB)
#LaserAlignmentEventFilter.DET_IDs.extend(LasDefs.DET_AT_TIB)
#LaserAlignmentEventFilter.DET_IDs.extend(LasDefs.DET_AT_TECp)
#LaserAlignmentEventFilter.DET_IDs.extend(LasDefs.DET_AT_TECm)


#LaserAlignmentEventFilter.SIGNAL_IDs.extend(LasDefs.SIGNAL_IDs_TECp_R4)
#LaserAlignmentEventFilter.SIGNAL_IDs.extend(LasDefs.SIGNAL_IDs_TECp_R6)
#LaserAlignmentEventFilter.SIGNAL_IDs.extend(LasDefs.SIGNAL_IDs_TECm_R4)
#LaserAlignmentEventFilter.SIGNAL_IDs.extend(LasDefs.SIGNAL_IDs_TECm_R6)
#LaserAlignmentEventFilter.SIGNAL_IDs.extend(LasDefs.SIGNAL_IDs_AT_TOB)
#LaserAlignmentEventFilter.SIGNAL_IDs.extend(LasDefs.SIGNAL_IDs_AT_TIB)
#LaserAlignmentEventFilter.SIGNAL_IDs.extend(LasDefs.SIGNAL_IDs_AT_TECp)
#LaserAlignmentEventFilter.SIGNAL_IDs.extend(LasDefs.SIGNAL_IDs_AT_TECm)

