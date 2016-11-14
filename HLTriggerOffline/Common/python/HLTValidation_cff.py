
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
hltassociation = cms.Sequence(
    hltMultiTrackValidation
    +hltMultiPVValidation
    +egammaSelectors
    +ExoticaValidationProdSeq
    )
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

hltvalidation = cms.Sequence(
    HLTMuonVal
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

# some hlt collections have no direct fastsim equivalent
# remove the dependent modules for now
# probably it would be rather easy to add or fake these collections
from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    hltassociation.remove(hltMultiTrackValidation)
    hltassociation.remove(hltMultiPVValidation)

hltvalidation_preprod = cms.Sequence(
  HLTTauVal
  +heavyFlavorValidationSequence
  +HLTSusyExoValSeq
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

    
