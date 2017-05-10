
from HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi import *
from HLTriggerOffline.Muon.HLTMuonPostVal_cff import *
from HLTriggerOffline.Egamma.EgammaPostProcessor_cfi import *
from HLTriggerOffline.Top.topHLTValidationHarvest_cff import *
from HLTriggerOffline.B2G.b2gHLTValidationHarvest_cff import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationHarvestingSequence_cff import *
from HLTriggerOffline.JetMET.Validation.JetMETPostProcessor_cff import *
#from HLTriggerOffline.special.hltAlCaPostVal_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_postProcessor_cff import *
from HLTriggerOffline.Higgs.HLTHiggsPostVal_cff import *
from HLTriggerOffline.Exotica.hltExoticaPostProcessors_cff import *
from HLTriggerOffline.SMP.HLTSMPPostVal_cff import *
from Validation.RecoTrack.HLTpostProcessorTracker_cfi import *
from Validation.RecoVertex.HLTpostProcessorVertex_cfi import *
#from HLTriggerOffline.Common.PostProcessorExample_cfi import *
from HLTriggerOffline.Common.HLTValidationQT_cff import *
from HLTriggerOffline.Btag.HltBtagPostValidation_cff import *
from HLTriggerOffline.Egamma.HLTpostProcessorGsfTracker_cfi import *

hltpostvalidation = cms.Sequence( 
    postProcessorHLTtrackingSequence
    +postProcessorHLTvertexing
    +HLTMuonPostVal
    +HLTTauPostVal
    +EgammaPostVal
    + postProcessorHLTgsfTrackingSequence
    +topHLTriggerValidationHarvest
    +heavyFlavorValidationHarvestingSequence
    +JetMETPostVal
    #+HLTAlCaPostVal
    +SusyExoPostVal
   #+ExamplePostVal
    +hltvalidationqt
    +HLTHiggsPostVal
    +hltExoticaPostProcessors
    +b2gHLTriggerValidationHarvest
    +HLTSMPPostVal
    +HltBTagPostVal
    )
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel

# fastsim customs
from Configuration.Eras.Modifier_fastSim_cff import fastSim
if fastSim.isChosen():
    hltpostvalidation.remove(postProcessorHLTtrackingSequence)
    hltpostvalidation.remove(postProcessorHLTvertexing)
    hltpostvalidation.remove(postProcessorHLTgsfTrackingSequence)
    # remove this:     +hltvalidationqt ?
    # remove this:    +hltExoticaPostProcessors ?
    
hltpostvalidation_preprod = cms.Sequence( 
    postProcessorHLTtrackingSequence
    +postProcessorHLTvertexing
    +HLTTauPostVal
    +heavyFlavorValidationHarvestingSequence
    +SusyExoPostVal
    + postProcessorHLTgsfTrackingSequence
   #+HLTHiggsPostVal
    )

hltpostvalidation_prod = cms.Sequence( 
    postProcessorHLTtrackingSequence
    +postProcessorHLTvertexing
    + postProcessorHLTgsfTrackingSequence
    )
