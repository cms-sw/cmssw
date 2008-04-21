import FWCore.ParameterSet.Config as cms

from L1Trigger.DTBti.dtbti_cff import *
from L1Trigger.DTTraco.dttraco_cff import *
from L1Trigger.DTTriggerServerTheta.dttstheta_cff import *
from L1Trigger.DTTriggerServerPhi.dttsphi_cff import *
TUParametersBlock = cms.PSet(
    TUParameters = cms.PSet(
        TSPhiParametersBlock,
        TracoParametersBlock,
        TSThetaParametersBlock,
        BtiParametersBlock,
        # MiniCrate setup time : fine syncronization
        SINCROTIME = cms.int32(0),
        # Debug flag
        Debug = cms.untracked.bool(False),
        # MiniCrate digi offset in tdc units
        DIGIOFFSET = cms.int32(500)
    )
)

