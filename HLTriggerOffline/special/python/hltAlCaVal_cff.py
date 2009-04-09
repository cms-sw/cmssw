from HLTriggerOffline.special.hltHITval_cfi import *
from HLTriggerOffline.special.MonitorAlCaEcalPi0_cfi import *
from HLTriggerOffline.special.EcalPhiSymHLTVal_cfi import *

HLTAlCaVal = cms.Sequence(
    EcalPi0Mon
    +EcalPhiSymMon
    +hltHITval
    )

import HLTriggerOffline.special.EcalPhiSymHLTVal_cfi
hltHITvalFastSim = HLTriggerOffline.special.hltHITval_cfi.hltHITval.clone()
hltHITvalFastSim.gtDigiLabel = cms.InputTag("gtDigis")
HLTAlCaVal_FastSim = cms.Sequence(
    EcalPi0Mon
    +EcalPhiSymMon
    +hltHITvalFastSim
    )


