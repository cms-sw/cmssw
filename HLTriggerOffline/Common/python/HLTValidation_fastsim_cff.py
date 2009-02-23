from HLTriggerOffline.Tau.Validation.HLTTauValidation_cff import *
from HLTriggerOffline.special.MonitorAlCaEcalPi0_cfi import *
from HLTriggerOffline.special.EcalPhiSymHLTVal_cfi import *
from HLTriggerOffline.Egamma.EgammaValidation_cff import *
hltvalidation = cms.Sequence(
    HLTTauVal
    +EcalPi0Mon
    +EcalPhiSymMon
    +egammaValidationSequence
    )
