from HLTriggerOffline.Muon.HLTMuonVal_cff import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cff import *
from HLTriggerOffline.Egamma.EgammaValidation_cff import *
from HLTriggerOffline.Top.topvalidation_cfi import *
from HLTriggerOffline.Common.FourVectorHLTriggerOffline_cff import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationSequence_cff import *
from HLTriggerOffline.JetMET.Validation.HLTJetMETValidation_cff import *
from HLTriggerOffline.special.hltAlCaVal_cff import *
from HLTriggerOffline.SUSYBSM.SusyExoValidation_cff import *
from HLTriggerOffline.Higgs.HiggsValidation_cff import *

# offline dqm:
# from DQMOffline.Trigger.DQMOffline_Trigger_cff.py import *
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *
#from DQMOffline.Trigger.MuonTrigRateAnalyzer_cfi import *
# online dqm:
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
    +HiggsValidationSequence
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
    +HiggsValidationSequence
    )

hltvalidation_preprod = cms.Sequence(
  HLTTauVal
  +HLTTopVal
  +HLTFourVector
  +heavyFlavorValidationSequence
  +HLTSusyExoValSeq
 #+HiggsValidationSequence
 )

hltvalidation_prod = cms.Sequence(
  HLTFourVector
  )

trigdqm_forValidation = cms.Sequence(
    hltMonTauReco+HLTTauDQMOffline
    +egHLTOffDQMSource
    )

hltvalidation_withDQM = cms.Sequence(
    hltvalidation
    +trigdqm_forValidation
    )
