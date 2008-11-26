from HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi import *
from HLTriggerOffline.Muon.PostProcessor_cfi import *
from HLTriggerOffline.Egamma.EgammaPostProcessor_cfi import *
from HLTriggerOffline.Top.PostProcessor_cfi import *
#from HLTriggerOffline.Common.PostProcessorExample_cfi import *
hltpostvalidation = cms.Sequence( 
    HLTMuonPostVal
    +HLTTauPostVal
    +EgammaPostVal
    +HLTTopPostVal
  #+ExamplePostVal
    )
