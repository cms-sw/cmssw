import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.tauSpinnerTableProducer_cfi import tauSpinnerTableProducer

tauSpinnerTable = tauSpinnerTableProducer.clone(
    src = 'prunedGenParticles',
    name = 'TauSpinner',
    theta = [0, 0.25, 0.5, -0.25, 0.375],
    pdfSet = 'NNPDF31_nnlo_hessian_pdfas',
    cmsE = 13600,
    defaultWeight = 1
)
(~run3_common).toModify(
    tauSpinnerTable, cmsE = 13000
)

tauSpinnerTableTask = cms.Task(tauSpinnerTable)
