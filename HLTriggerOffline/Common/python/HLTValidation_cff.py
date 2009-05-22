from HLTriggerOffline.Muon.HLTMuonVal_cff import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cff import *
from HLTriggerOffline.Egamma.EgammaValidation_cff import *
from HLTriggerOffline.Top.topvalidation_cfi import *
from HLTriggerOffline.Common.FourVectorHLTriggerOffline_cff import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationSequence_cff import *
from HLTriggerOffline.JetMET.Validation.HLTJetMETValidation_cff import *
from HLTriggerOffline.special.hltAlCaVal_cff import *
from HLTriggerOffline.SUSYBSM.SusyExoValidation_cff import *

# from DQMOffline.Trigger.DQMOffline_Trigger_cff.py import *
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *
from DQMOffline.Trigger.MuonTrigRateAnalyzer_cfi import *

# These are what you need for Online DQM
from DQM.HLTEvF.HLTMonTau_cfi import *

hltvalidation = cms.Sequence(
    HLTMuonVal
    +HLTTauVal
    +egammaValidationSequence
    +HLTTopVal
    +HLTFourVector
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    +HLTAlCaVal
    +HLTSusyExoValSeq
    )

#Muon offline DQM not ready yet.  Tag is in preparation
hltvalidation_withDQM = cms.Sequence(
     HLTMuonVal
    +HLTTauVal+hltMonTauReco+HLTTauDQMOffline
    +egammaValidationSequence+egHLTOffDQMSource
    +HLTTopVal
    +HLTFourVector
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    +HLTAlCaVal
    +HLTSusyExoValSeq
    )

hltvalidation_fastsim = cms.Sequence(
     HLTMuonVal_FastSim
    +HLTTauVal
    +egammaValidationSequence
    +HLTTopVal
    +HLTFourVector
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    +HLTAlCaVal_FastSim
    +HLTSusyExoValSeq_FastSim
    )

hltvalidation_pu = cms.Sequence(
     HLTMuonVal
    +HLTTauVal
    +egammaValidationSequence
    +HLTTopVal
    +HLTFourVector
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    +HLTAlCaVal
    +HLTSusyExoValSeq
    )
