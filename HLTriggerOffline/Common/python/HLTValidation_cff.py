from Validation.RecoTrack.HLTmultiTrackValidator_cff import *
from Validation.RecoVertex.HLTmultiPVvalidator_cff import *
from HLTriggerOffline.Muon.HLTMuonVal_cff import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cff import *
from HLTriggerOffline.Egamma.EgammaValidationAutoConf_cff import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationSequence_cff import *
from HLTriggerOffline.JetMET.Validation.HLTJetMETValidation_cff import *
#from HLTriggerOffline.special.hltAlCaVal_cff import *
from HLTriggerOffline.SUSYBSM.SusyExoValidation_cff import *
from HLTriggerOffline.Higgs.HiggsValidation_cff import *
from HLTriggerOffline.Top.topHLTValidation_cff import *
from HLTriggerOffline.B2G.b2gHLTValidation_cff import *
from HLTriggerOffline.Exotica.ExoticaValidation_cff import *
from HLTriggerOffline.SMP.SMPValidation_cff import *
from HLTriggerOffline.Btag.HltBtagValidation_cff import *
from HLTriggerOffline.Btag.HltBtagValidationFastSim_cff import  *

# offline dqm:
# from DQMOffline.Trigger.DQMOffline_Trigger_cff.py import *
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *
from DQMOffline.Trigger.topHLTOfflineDQM_cff import *
#from DQMOffline.Trigger.MuonTrigRateAnalyzer_cfi import *
# online dqm:
from DQMOffline.Trigger.HLTMonTau_cfi import *
 
# additional producer sequence prior to hltvalidation
# to evacuate producers/filters from the EndPath
hltassociation = cms.Sequence( egammaSelectors
                               +ExoticaValidationProdSeq )


hltvalidation = cms.Sequence(
    hltMultiTrackValidation
    +hltMultiPVValidation
    +HLTMuonVal
    +HLTTauVal
    +egammaValidationSequence
    +topHLTriggerOfflineDQM
    +topHLTriggerValidation
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    +HLTSusyExoValSeq
    +HiggsValidationSequence
    +ExoticaValidationSequence
    +b2gHLTriggerValidation
    +SMPValidationSequence
    +hltbtagValidationSequence
    )


# additional producer sequence prior to hltvalidation_fastsim
# to evacuate producers from the EndPath
hltassociation_fastsim = cms.Sequence(
    HLTMuonAss_FastSim
  + egammaSelectors
  + hltTauRef
)

hltvalidation_fastsim = cms.Sequence(
     HLTMuonVal_FastSim
    +HLTTauValFS
    +egammaValidationSequenceFS
    +topHLTriggerOfflineDQM
    +topHLTriggerValidation
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    #+HLTAlCaVal_FastSim
    +HLTSusyExoValSeq_FastSim
    +HiggsValidationSequence
    +b2gHLTriggerValidation
    +SMPValidationSequence
    +hltbtagValidationSequenceFastSim
    )

hltvalidation_preprod = cms.Sequence(
  HLTTauVal
  +heavyFlavorValidationSequence
  +HLTSusyExoValSeq
 #+HiggsValidationSequence
 )

hltvalidation_preprod_fastsim = cms.Sequence(
 HLTTauVal
 +heavyFlavorValidationSequence
 +HLTSusyExoValSeq_FastSim
#+HiggsValidationSequence
)

hltvalidation_prod = cms.Sequence(
  )

trigdqm_forValidation = cms.Sequence(
    hltMonTauReco+HLTTauDQMOffline
    +egHLTOffDQMSource
    )

hltvalidation_withDQM = cms.Sequence(
    hltvalidation
    +trigdqm_forValidation
    )
