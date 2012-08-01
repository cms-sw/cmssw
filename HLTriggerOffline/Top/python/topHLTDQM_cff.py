import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Top.topDiLeptonOfflineDQM_cfi import *
from HLTriggerOffline.Top.topSingleLeptonHLTEventDQM_cfi import *
from HLTriggerOffline.Top.singletopHLTEventDQM_cfi import *
from HLTriggerOffline.Top.topvalidation_cfi import *
from JetMETCorrections.Configuration.JetCorrectionProducersAllAlgos_cff import *



topHLTDQM = cms.Sequence(  
                           topDiLeptonOfflineDQM
                           *topSingleMuonMediumTriggerDQM
                           *topSingleElectronMediumTriggerDQM
			   *SingleTopSingleMuonTriggerDQM
			   *SingleTopSingleElectronTriggerDQM	
                           )

