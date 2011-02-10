import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigBti_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigTraco_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigLUTs_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigTSTheta_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigTSPhi_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigTU_cff import *
from L1TriggerConfig.DTTPGConfigProducers.L1DTConfigSectColl_cff import *

DTTPGParametersBlock = cms.PSet(
    DTTPGParameters = cms.PSet(
        SectCollParametersBlock,
        Debug = cms.untracked.bool(False),
        TUParameters = cms.PSet(
            TracoParametersBlock,
            TSPhiParametersBlock,
            TSThetaParametersBlock,
            TUParamsBlock,
            BtiParametersBlock,
            LutParametersBlock
        )
    )
)


