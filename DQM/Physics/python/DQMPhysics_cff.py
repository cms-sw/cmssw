import FWCore.ParameterSet.Config as cms

from DQM.Physics.bphysicsOniaDQM_cfi import *
from DQM.Physics.ewkMuDQM_cfi import *
from DQM.Physics.ewkElecDQM_cfi import *
from DQM.Physics.ewkMuLumiMonitorDQM_cfi import *
from DQM.Physics.qcdPhotonsDQM_cfi import *
from DQM.Physics.topDiLeptonOfflineDQM_cfi import *
from DQM.Physics.topSingleLeptonDQM_cfi import *
from DQM.Physics.topSingleLeptonDQM_PU_cfi import *
from DQM.Physics.singleTopDQM_cfi import *
from DQM.Physics.ewkMuLumiMonitorDQM_cfi import *
from DQM.Physics.susyDQM_cfi import *
from DQM.Physics.HiggsDQM_cfi import *
from JetMETCorrections.Configuration.JetCorrectionProducersAllAlgos_cff import *

## unused
#from DQM.Physics.qcdUeDQM_cfi import *
#from DQM.Physics.ewkTauDQM_cfi import *
#from DQM.Physics.qcdHighPtDQM_cfi import *
#from DQM.Physics.ewkDQM_cfi import *


dqmPhysics = cms.Sequence( bphysicsOniaDQM 
                           *ewkMuDQM
                           *ewkElecDQM
                           *ewkMuLumiMonitorDQM
                           *qcdPhotonsDQM
                           *topDiLeptonOfflineDQM
                           *topSingleLeptonDQM
                           *singleTopDQM
                           *susyDQM
                           *HiggsDQM
#                           *ewkDQM
#                           *qcdHighPtDQM
#                           *ewkTauDQM
#                           *QcdUeDQM                           
                           )

bphysicsOniaDQMHI = bphysicsOniaDQM.clone(vertex=cms.InputTag("hiSelectedVertex"))
dqmPhysicsHI = cms.Sequence(bphysicsOniaDQMHI)

from DQM.Physics.qcdPhotonsCosmicDQM_cff import *
dqmPhysicsCosmics = cms.Sequence(dqmPhysics)
dqmPhysicsCosmics.replace(qcdPhotonsDQM, qcdPhotonsCosmicDQM)
