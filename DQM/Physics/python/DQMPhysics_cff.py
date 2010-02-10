import FWCore.ParameterSet.Config as cms

from DQM.Physics.bphysicsOniaDQM_cfi import *
from DQM.Physics.ewkDQM_cfi import *
from DQM.Physics.ewkMuDQM_cfi import *
from DQM.Physics.ewkElecDQM_cfi import *
from DQM.Physics.ewkMuLumiMonitorDQM_cfi import *
from DQM.Physics.qcdPhotonsDQM_cfi import *
from DQM.Physics.qcdHighPtDQM_cfi import *
from DQM.Physics.topDiLeptonDQM_cfi import *
from DQM.Physics.ewkMuLumiMonitorDQM_cfi import *
from DQM.Physics.ewkTauDQM_cfi import *

dqmPhysics = cms.Sequence(bphysicsOniaDQM
			 *ewkDQM
                         *ewkMuDQM
                         *ewkElecDQM
                         *ewkMuLumiMonitorDQM
			 *qcdPhotonsDQM
                         *qcdHighPtDQM
			 *topDiLeptonDQM
                         *ewkTauDQM
			 )

