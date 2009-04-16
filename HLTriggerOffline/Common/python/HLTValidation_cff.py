from HLTriggerOffline.Muon.HLTMuonVal_cff import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cff import *
from HLTriggerOffline.Egamma.EgammaValidation_cff import *
from HLTriggerOffline.Top.topvalidation_cfi import *
from HLTriggerOffline.Common.FourVectorHLTriggerOffline_cff import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationSequence_cff import *
from HLTriggerOffline.JetMET.Validation.HLTJetMETValidation_cff import *
from HLTriggerOffline.special.hltAlCaVal_cff import *
#from HLTriggerOffline.SUSYBSM.SusyExoValidation_cff import *

hltvalidation = cms.Sequence(
     HLTMuonVal
    +HLTTauVal
    +egammaValidationSequence
    +HLTTopVal
    +HLTFourVector
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    +HLTAlCaVal
   #+HLTSusyExoValSeq
    )

hltvalidation_fastsim = cms.Sequence(
   # HLTMuonVal_FastSim
     HLTTauVal
    +egammaValidationSequence
    +HLTTopVal
    +HLTFourVector
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    +HLTAlCaVal_FastSim
   #+HLTSusyExoValSeq_FastSim
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
   #+HLTSusyExoValSeq
    )
