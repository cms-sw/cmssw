import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.B2G.topDiLeptonHLTEventValidation_cfi import *
from HLTriggerOffline.B2G.topSingleLeptonHLTEventValidation_cfi import *
from HLTriggerOffline.B2G.singletopHLTEventValidation_cfi import *

b2gHLTriggerValidation = cms.Sequence(  
    topSingleMuonHLTValidation
    *topSingleElectronHLTValidation
    )

