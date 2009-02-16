from HLTriggerOffline.Muon.muonTriggerRateTimeAnalyzer_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cff import *
from HLTriggerOffline.special.MonitorAlCaEcalPi0_cfi import *
from HLTriggerOffline.special.EcalPhiSymHLTVal_cfi import *
from HLTriggerOffline.Egamma.EgammaValidation_cff import *
from HLTriggerOffline.Top.topvalidation_cfi import *
#from HLTriggerOffline.SUSYBSM.SUSYBSM_triggerValidation_cff import *
from HLTriggerOffline.Common.FourVectorHLTriggerOffline_cff import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationSequence_cff import *
hltvalidation = cms.Sequence(
    muonTriggerRateTimeAnalyzer
    +HLTTauVal
    +EcalPi0Mon
    +EcalPhiSymMon
    +egammaValidationSequence
    +HLTTopVal
   #+HLTSusyExoVal
    +HLTFourVector
    +heavyFlavorValidationSequence
    )

