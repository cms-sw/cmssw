import FWCore.ParameterSet.Config as cms

# FastSim version of DQMOffline/Configuration/python/DQMOfflineMC_cff.py .
# On long term FastSim should actually use DQMOffline/Configuration/python/DQMOfflineMC_cff.py, 
# with fastsim modifications applied via fastSim era.
# For now this is too much of a hassle because DQMOffline/Configuration/python/DQMOfflineMC_cff.py is a bit of a mess.
# Therefore we define here a FastSim specific DQM sequence with only the most relevant modules, and/or the ones that don't cause too much headache

from DQMOffline.RecoB.PrimaryVertexMonitor_cff import *
from DQM.Physics.DQMPhysics_cff import *
from Validation.RecoTau.DQMSequences_cfi import *
from DQMOffline.RecoB.dqmAnalyzer_cff import *

DQMOfflinePrePOG = cms.Sequence(
    pvMonitor *
    bTagPlotsDATA *
    dqmPhysics *
    produceDenoms *
    pfTauRunDQMValidation
    
    )

DQMOffline = cms.Sequence(DQMOfflinePrePOG)
PostDQMOffline = cms.Sequence()
