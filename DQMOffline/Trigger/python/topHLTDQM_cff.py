import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.topDiLeptonHLTEventDQM_cfi import *
#from DQMOffline.Trigger.topSingleLeptonHLTEventDQM_cfi import *
#from DQMOffline.Trigger.singletopHLTEventDQM_cfi import *
from JetMETCorrections.Configuration.JetCorrectionProducersAllAlgos_cff import *



topHLTriggerDQM = cms.Sequence(  
        DiMuonDQM
        *DiElectronDQM
        *ElecMuonDQM
        #*topSingleMuonMediumTriggerDQM
        #*topSingleElectronMediumTriggerDQM
        #*SingleTopSingleMuonTriggerDQM
        #*SingleTopSingleElectronTriggerDQM	
        )

