from HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi import *
from HLTriggerOffline.Muon.HLTMuonPostVal_cff import *
from HLTriggerOffline.Egamma.EgammaPostProcessor_cfi import *
from HLTriggerOffline.Top.PostProcessor_cfi import *
from HLTriggerOffline.Common.FourVectorHLTriggerOfflineClient_cfi import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationHarvestingSequence_cff import *
from HLTriggerOffline.JetMET.Validation.JetMETPostProcessor_cff import *
from HLTriggerOffline.special.hltAlCaPostVal_cff import *
#from HLTriggerOffline.SUSYBSM.SUSYBSM_postProcessor_cff import *
#from HLTriggerOffline.Common.PostProcessorExample_cfi import *

hltpostvalidation = cms.Sequence( 
    #HLTMuonPostVal
     HLTTauPostVal
   #+EgammaPostVal
    +HLTTopPostVal
    +HLTriggerOfflineFourVectorClient
    +heavyFlavorValidationHarvestingSequence
    +JetMETPostVal
    +HLTAlCaPostVal
   #+SusyExoPostVal
   #+ExamplePostVal
    )

hltpostvalidation_fastsim = cms.Sequence( 
    #HLTMuonPostVal_FastSim
     HLTTauPostVal
   #+EgammaPostVal
    +HLTriggerOfflineFourVectorClient
    +HLTTopPostVal
    +heavyFlavorValidationHarvestingSequence
    +JetMETPostVal
    #+HLTAlCaPostVal
    #+SusyExoPostVal
    )

hltpostvalidation_pu = cms.Sequence( 
    #HLTMuonPostVal
     HLTTauPostVal
   #+EgammaPostVal
    +HLTriggerOfflineFourVectorClient
    +HLTTopPostVal
    +heavyFlavorValidationHarvestingSequence
    +JetMETPostVal
    +HLTAlCaPostVal
   #+SusyExoPostVal
    )
