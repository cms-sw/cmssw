from HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi import *
from HLTriggerOffline.Muon.HLTMuonPostVal_cff import *
from HLTriggerOffline.Egamma.EgammaPostProcessor_cfi import *
from HLTriggerOffline.Top.topHLTValidationHarvest_cff import *
from HLTriggerOffline.Common.FourVectorHLTriggerOfflineClient_cfi import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationHarvestingSequence_cff import *
from HLTriggerOffline.JetMET.Validation.JetMETPostProcessor_cff import *
#from HLTriggerOffline.special.hltAlCaPostVal_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_postProcessor_cff import *
from HLTriggerOffline.Higgs.HLTHiggsPostVal_cff import *
from Validation.RecoTrack.HLTpostProcessorTracker_cfi import *

#from HLTriggerOffline.Common.PostProcessorExample_cfi import *
from HLTriggerOffline.Common.HLTValidationQT_cff import *

hltpostvalidation = cms.Sequence( 
    postProcessorHLTtracking
     +HLTMuonPostVal
    +HLTTauPostVal
    +EgammaPostVal
    +topHLTriggerValidationHarvest
    +hltriggerFourVectorClient
    +heavyFlavorValidationHarvestingSequence
    +JetMETPostVal
    #+HLTAlCaPostVal
    +SusyExoPostVal
   #+ExamplePostVal
    +hltvalidationqt
    +HLTHiggsPostVal
    )

hltpostvalidation_fastsim = cms.Sequence( 
     HLTMuonPostVal_FastSim
    +HLTTauPostVal
    +EgammaPostVal
    +hltriggerFourVectorClient
    +topHLTriggerValidationHarvest
    +heavyFlavorValidationHarvestingSequence
    +JetMETPostVal
    #+HLTAlCaPostVal
    +SusyExoPostVal
    +HLTHiggsPostVal
    )

hltpostvalidation_preprod = cms.Sequence( 
    postProcessorHLTtracking
    +HLTTauPostVal
    +hltriggerFourVectorClient
    +heavyFlavorValidationHarvestingSequence
    +SusyExoPostVal
   #+HLTHiggsPostVal
    )

hltpostvalidation_prod = cms.Sequence( 
    postProcessorHLTtracking
    +hltriggerFourVectorClient
    )
