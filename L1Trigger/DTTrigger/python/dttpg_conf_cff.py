import FWCore.ParameterSet.Config as cms

from L1Trigger.DTSectorCollector.dtsectcoll_cff import *
from L1Trigger.DTSectorCollector.dttu_cff import *
DTTPGParametersBlock = cms.PSet(
    DTTPGParameters = cms.PSet(
        SectCollParametersBlock,
        TUParametersBlock,
        Debug = cms.untracked.bool(False)
    )
)

