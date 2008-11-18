from HLTriggerOffline.Muon.muonTriggerRateTimeAnalyzer_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cff import *
from HLTriggerOffline.special.MonitorAlCaEcalPi0_cfi import *
hltvalidation = cms.Sequence(
    muonTriggerRateTimeAnalyzer
    *HLTTauVal
    *EcalPi0Mon
    )
