from DQM.HLTEvF.HLTAlCaMonPi0_cfi import *
from DQM.HLTEvF.HLTAlCaMonEcalPhiSym_cfi import *

HLTAlCaMon = cms.Sequence(
    EcalPi0Mon
    +EcalPhiSymMon
    )
