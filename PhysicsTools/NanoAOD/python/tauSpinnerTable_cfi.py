import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.tauSpinnerTableProducer_cfi import tauSpinnerTableProducer

tauSpinnerTable = tauSpinnerTableProducer.clone(
    src = 'prunedGenParticles',
    name = 'TauSpinner',
    theta = [0, 0.25, 0.5, -0.25, 0.375],
    defaultWeight = 1
)

tauSpinnerTableTask = cms.Task(tauSpinnerTable)
