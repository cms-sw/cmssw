import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.B2G.b2gSingleLeptonHLTEventValidation_cfi import *
from HLTriggerOffline.B2G.b2gHadronicHLTEventValidation_cfi import *


b2gHLTriggerValidation = cms.Sequence(  
    b2gSingleMuonHLTValidation*
    b2gSingleElectronHLTValidation*
    b2gElePlusSingleJetHLTValidation*
    b2gSingleJetHLTValidation*
    b2gDiJetHLTValidation
    )

