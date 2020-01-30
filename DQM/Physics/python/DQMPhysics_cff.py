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
from DQM.Physics.CentralityDQM_cfi import *
from DQM.Physics.CentralitypADQM_cfi import *
from DQM.Physics.topJetCorrectionHelper_cfi import *
from DQM.Physics.FSQDQM_cfi import *
from DQM.Physics.SMPDQM_cfi import *

dqmPhysics = cms.Sequence( bphysicsOniaDQM 
#                          Empty Histograms from EWK and Error log messages
#                           *ewkMuDQM	   # errors in logs due HLTPrescaleProvider
#                           *ewkElecDQM    # Error in determining HLT prescale set index from L1 data using L1GtUtils 
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
                           *FSQDQM
                           *SMPDQM
                           )

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toReplaceWith(dqmPhysics, dqmPhysics.copyAndExclude([ # FIXME
    ewkMuDQM,            # Excessive printouts because 2017 doesn't have HLT yet
    ewkElecDQM,          # Excessive printouts because 2017 doesn't have HLT yet
    ewkMuLumiMonitorDQM, # Excessive printouts because 2017 doesn't have HLT yet
]))
from Configuration.Eras.Modifier_pA_2016_cff import pA_2016
dqmPhysicspA  =  dqmPhysics.copy()
dqmPhysicspA += CentralitypADQM
pA_2016.toReplaceWith(dqmPhysics, dqmPhysicspA)
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
_dqmPhysics  =  dqmPhysics.copy()
_dqmPhysics += CentralityDQM
pp_on_AA_2018.toModify(CentralityDQM, vertexcollection=cms.InputTag("offlinePrimaryVertices"))
pp_on_AA_2018.toReplaceWith(dqmPhysics, _dqmPhysics)

bphysicsOniaDQMHI = bphysicsOniaDQM.clone(vertex=cms.InputTag("hiSelectedVertex"))
dqmPhysicsHI = cms.Sequence(bphysicsOniaDQMHI+CentralityDQM)

from DQM.Physics.qcdPhotonsCosmicDQM_cff import *
dqmPhysicsCosmics = cms.Sequence(dqmPhysics)
dqmPhysicsCosmics.replace(qcdPhotonsDQM, qcdPhotonsCosmicDQM)
