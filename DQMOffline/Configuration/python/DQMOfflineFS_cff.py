import FWCore.ParameterSet.Config as cms

# FastSim version of DQMOffline/Configuration/python/DQMOfflineMC_cff.py .

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
PostDQMOffline = cms.Sequence()
