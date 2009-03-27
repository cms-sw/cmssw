from HLTriggerOffline.special.hltHITval_cfi import *
from HLTriggerOffline.special.MonitorAlCaEcalPi0_cfi import *
from HLTriggerOffline.special.EcalPhiSymHLTVal_cfi import *
HLTAlCaVal = cms.Sequence(
    EcalPi0Mon
    +EcalPhiSymMon
    +hltHITval
    )

