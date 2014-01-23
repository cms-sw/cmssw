import FWCore.ParameterSet.Config as cms

#from HLTriggerOffline.Top.topDiLeptonHLTEventValidation_cfi import *
from HLTriggerOffline.Top.topSingleLeptonHLTEventValidation_cfi import *
from HLTriggerOffline.Top.singletopHLTEventValidation_cfi import *

topHLTriggerValidation = cms.Sequence(  
#        DiMuonHLTValidation
#        *DiElectronHLTValidation
#        *ElecMuonHLTValidation
#        *
        topSingleMuonHLTValidation
        *topSingleElectronHLTValidation
        *SingleTopSingleMuonHLTValidation
        *SingleTopSingleElectronHLTValidation	
        )

