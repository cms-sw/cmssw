import FWCore.ParameterSet.Config as cms

from DQM.Physics.bphysicsOniaDQM_cfi import *
from DQM.Physics.ewkMuDQM_cfi import *
from DQM.Physics.ewkElecDQM_cfi import *
from DQM.Physics.ewkMuLumiMonitorDQM_cfi import *
from DQM.Physics.qcdPhotonsDQM_cfi import *
from DQM.Physics.topSingleLeptonDQM_cfi import *
from DQM.Physics.topDiLeptonOfflineDQM_cfi import *
from DQM.Physics.topSingleLeptonDQM_PU_cfi import *
from DQM.Physics.singleTopDQM_cfi import *
from DQM.Physics.ewkMuLumiMonitorDQM_cfi import *
from DQM.Physics.susyDQM_cfi import *
from DQM.Physics.HiggsDQM_cfi import *
from DQM.Physics.ExoticaDQM_cfi import *
from DQM.Physics.B2GDQM_cfi import *
from DQM.PhysicsHWW.hwwDQM_cfi import *

#############################################################################
## Temporary due to bad naming of the jet algorithm in correction modules  ##
from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFCHSL1Offset, ak4PFCHSL1Fastjet, ak4PFCHSL2Relative, ak4PFCHSL3Absolute, ak4PFCHSResidual, ak4PFCHSL2L3, ak4PFCHSL2L3Residual
ak4PFCHSL1Offset.algorithm = 'AK5PFchs'
ak4PFCHSL1Fastjet.algorithm = 'AK5PFchs'
ak4PFCHSL2Relative.algorithm = 'AK5PFchs'
ak4PFCHSL3Absolute.algorithm = 'AK5PFchs'
ak4PFCHSResidual.algorithm = 'AK5PFchs'

topDQMak5PFCHSL1Offset = ak4PFCHSL1Offset.clone()
topDQMak5PFCHSL1Fastjet = ak4PFCHSL1Fastjet.clone()
topDQMak5PFCHSL2Relative = ak4PFCHSL2Relative.clone()
topDQMak5PFCHSL3Absolute = ak4PFCHSL3Absolute.clone()
topDQMak5PFCHSResidual = ak4PFCHSResidual.clone()

topDQMak5PFCHSL2L3 = ak4PFCHSL2L3.clone(correctors = cms.vstring('topDQMak5PFCHSL2Relative','topDQMak5PFCHSL3Absolute'))
topDQMak5PFCHSL2L3Residual = ak4PFCHSL2L3Residual.clone(correctors = cms.vstring('topDQMak5PFCHSL2Relative','topDQMak5PFCHSL3Absolute','topDQMak5PFCHSResidual'))
#############################################################################

dqmPhysics = cms.Sequence( bphysicsOniaDQM 
                           *ewkMuDQM
                           *ewkElecDQM
                           *ewkMuLumiMonitorDQM
                           *qcdPhotonsDQM
			   *topSingleMuonMediumDQM
                           *topSingleElectronMediumDQM	
                           *singleTopMuonMediumDQM
                           *singleTopElectronMediumDQM
                           *DiMuonDQM
			   *DiElectronDQM
			   *ElecMuonDQM
                           *susyDQM
                           *HiggsDQM
                           *ExoticaDQM
                           *B2GDQM
                           *hwwDQM
                           )

bphysicsOniaDQMHI = bphysicsOniaDQM.clone(vertex=cms.InputTag("hiSelectedVertex"))
dqmPhysicsHI = cms.Sequence(bphysicsOniaDQMHI)

from DQM.Physics.qcdPhotonsCosmicDQM_cff import *
dqmPhysicsCosmics = cms.Sequence(dqmPhysics)
dqmPhysicsCosmics.replace(qcdPhotonsDQM, qcdPhotonsCosmicDQM)
dqmPhysicsCosmics.replace(hwwDQM, hwwCosmicDQM)
