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
from HLTriggerOffline.B2G.b2gHLTValidation_cff import *
from HLTriggerOffline.Exotica.ExoticaValidation_cff import *
from HLTriggerOffline.SMP.SMPValidation_cff import *
from HLTriggerOffline.Btag.HltBtagValidation_cff import *
from HLTriggerOffline.Egamma.HLTmultiTrackValidatorGsfTracks_cff import *
from HLTriggerOffline.Muon.HLTmultiTrackValidatorMuonTracks_cff import *
# HCAL
from Validation.HcalDigis.HLTHcalDigisParam_cfi import *
from Validation.HcalRecHits.HLTHcalRecHitParam_cfi import *

# offline dqm:
# from DQMOffline.Trigger.DQMOffline_Trigger_cff.py import *
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *
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
    +hltMultiTrackValidationGsfTracks
    +hltMultiTrackValidationMuonTracks
    )
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

# Temporary Phase-2 config
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(hltassociation, hltassociation.copyAndExclude([hltMultiTrackValidation,
                                                                           hltMultiPVValidation,
                                                                           egammaSelectors,
                                                                           ExoticaValidationProdSeq,
                                                                           hltMultiTrackValidationGsfTracks,
                                                                           hltMultiTrackValidationMuonTracks])
)

# hcal
from DQMOffline.Trigger.HCALMonitoring_cff import *

hltvalidationCommon = cms.Sequence(
    hcalMonitoringSequence
)

hltvalidationWithMC = cms.Sequence(
    HLTMuonVal
    +HLTTauVal
    +egammaValidationSequence
    +heavyFlavorValidationSequence
    +HLTJetMETValSeq
    +HLTSusyExoValSeq
    +HiggsValidationSequence
    +ExoticaValidationSequence
    +b2gHLTriggerValidation
    +SMPValidationSequence
    +hltbtagValidationSequence #too noisy for now
    +hltHCALdigisAnalyzer+hltHCALRecoAnalyzer+hltHCALNoiseRates # HCAL
)

# Temporary Phase-2 config
# Exclude everything except JetMET for now
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(hltvalidationWithMC, hltvalidationWithMC.copyAndExclude([#HLTMuonVal,
                                                                                     HLTTauVal,
                                                                                     egammaValidationSequence,
                                                                                     heavyFlavorValidationSequence,
                                                                                     #HLTJetMETValSeq,
                                                                                     HLTSusyExoValSeq,
                                                                                     HiggsValidationSequence,
                                                                                     ExoticaValidationSequence,
                                                                                     b2gHLTriggerValidation,
                                                                                     SMPValidationSequence,
                                                                                     hltbtagValidationSequence,
                                                                                     hltHCALdigisAnalyzer,
                                                                                     hltHCALRecoAnalyzer,
                                                                                     hltHCALNoiseRates]))

hltvalidationWithData = cms.Sequence(
)

hltvalidation = cms.Sequence(
    hltvalidationCommon *
    hltvalidationWithMC *
    hltvalidationWithData
)

# some hlt collections have no direct fastsim equivalent
# remove the dependent modules for now
# probably it would be rather easy to add or fake these collections
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(hltassociation, hltassociation.copyAndExclude([
    hltMultiTrackValidation,
    hltMultiPVValidation,
    hltMultiTrackValidationGsfTracks,
    hltMultiTrackValidationMuonTracks,
]))

from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
for e in [pp_on_XeXe_2017, pp_on_AA]:
    e.toReplaceWith(hltvalidation, hltvalidation.copyAndExclude([HiggsValidationSequence]))

hltvalidation_preprod = cms.Sequence(
  HLTTauVal
  +heavyFlavorValidationSequence
  +HLTSusyExoValSeq
# +HiggsValidationSequence
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

    
