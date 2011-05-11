from HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi import *
from HLTriggerOffline.Muon.HLTMuonPostVal_cff import *
from HLTriggerOffline.Egamma.EgammaPostProcessor_cfi import *
from HLTriggerOffline.Top.HLTTopPostVal_cff import *
from HLTriggerOffline.Common.FourVectorHLTriggerOfflineClient_cfi import *
from HLTriggerOffline.HeavyFlavor.heavyFlavorValidationHarvestingSequence_cff import *
from HLTriggerOffline.JetMET.Validation.JetMETPostProcessor_cff import *
from HLTriggerOffline.special.hltAlCaPostVal_cff import *
from HLTriggerOffline.SUSYBSM.SUSYBSM_postProcessor_cff import *
from HLTriggerOffline.Higgs.HLTHiggsPostVal_cff import *

#from HLTriggerOffline.Common.PostProcessorExample_cfi import *
from HLTriggerOffline.Common.HLTValidationQT_cff import *

hltpostvalidation = cms.Sequence( 
     HLTMuonPostVal
    +HLTTauPostVal
    +EgammaPostVal
    +HLTTopPostVal
    +hltriggerFourVectorClient
    +heavyFlavorValidationHarvestingSequence
    +JetMETPostVal
    +HLTAlCaPostVal
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
    +HLTTopPostVal
    +heavyFlavorValidationHarvestingSequence
    +JetMETPostVal
    +HLTAlCaPostVal
    +SusyExoPostVal
    +HLTHiggsPostVal
    )

hltpostvalidation_preprod = cms.Sequence( 
    HLTTauPostVal
    +HLTTopPostVal
    +hltriggerFourVectorClient
    +heavyFlavorValidationHarvestingSequence
    +SusyExoPostVal
   #+HLTHiggsPostVal
    )

hltpostvalidation_prod = cms.Sequence( 
    hltriggerFourVectorClient
    )
