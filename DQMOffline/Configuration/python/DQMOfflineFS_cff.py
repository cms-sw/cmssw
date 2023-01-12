import FWCore.ParameterSet.Config as cms

# FastSim version of DQMOffline/Configuration/python/DQMOfflineMC_cff.py .
# MiniAOD part is a clone from DQMOffline/Configuration/python/DQMOffline_cff.py

from DQMOffline.RecoB.PrimaryVertexMonitor_cff import *
from DQM.Physics.DQMPhysics_cff import *
from Validation.RecoTau.DQMSequences_cfi import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *

DQMOfflinePrePOG = cms.Sequence(
    pvMonitor *
    bTagPlotsDATA *
    dqmPhysics *
    produceDenomsData *
    pfTauRunDQMValidation
    )

# Fix Jet Corrector in FastSim
topSingleMuonMediumDQM.setup.jetExtras.jetCorrector = cms.InputTag("ak4PFCHSL1FastL2L3Corrector")
topSingleElectronMediumDQM.setup.jetExtras.jetCorrector = cms.InputTag("ak4PFCHSL1FastL2L3Corrector")
singleTopMuonMediumDQM.setup.jetExtras.jetCorrector = cms.InputTag("ak4PFCHSL1FastL2L3Corrector")
singleTopElectronMediumDQM.setup.jetExtras.jetCorrector = cms.InputTag("ak4PFCHSL1FastL2L3Corrector")

DQMOfflineFS = cms.Sequence(DQMOfflinePrePOG)

# miniAOD DQM validation
from DQMOffline.JetMET.jetMETDQMOfflineSource_cff import *
from DQMOffline.Muon.muonMonitors_cff import *
from Validation.RecoParticleFlow.miniAODDQM_cff import * # On MiniAOD vs RECO
from Validation.RecoParticleFlow.DQMForPF_MiniAOD_cff import * # MiniAOD PF variables
from DQM.TrackingMonitor.tracksDQMMiniAOD_cff import *
from DQMOffline.RecoB.bTagMiniDQM_cff import *
from DQMOffline.Muon.miniAOD_cff import *
from DQM.Physics.DQMTopMiniAOD_cff import *

DQMOfflineMiniAOD = cms.Sequence(jetMETDQMOfflineRedoProductsMiniAOD*bTagMiniDQMSource*muonMonitors_miniAOD*MuonMiniAOD*DQMOfflinePF)

#Post sequences are automatically placed in the EndPath by ConfigBuilder if PAT is run.
#miniAOD DQM sequences need to access the filter results.

PostDQMOfflineMiniAOD = cms.Sequence(miniAODDQMSequence*jetMETDQMOfflineSourceMiniAOD*tracksDQMMiniAOD*topPhysicsminiAOD)
PostDQMOffline = cms.Sequence()

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toReplaceWith( PostDQMOfflineMiniAOD, PostDQMOfflineMiniAOD.copyAndExclude([
    pfMetDQMAnalyzerMiniAOD, pfPuppiMetDQMAnalyzerMiniAOD # No hcalnoise (yet)
]))

from PhysicsTools.NanoAOD.nanoDQM_cff import nanoDQM
DQMOfflineNanoAOD = cms.Sequence(nanoDQM)
#PostDQMOfflineNanoAOD = cms.Sequence(nanoDQM)
from PhysicsTools.NanoAOD.nanogenDQM_cff import nanogenDQM
DQMOfflineNanoGen = cms.Sequence(nanogenDQM)
