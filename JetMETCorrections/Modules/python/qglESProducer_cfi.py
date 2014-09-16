import FWCore.ParameterSet.Config as cms
QGLikelihoodESProducer_AK4PF = cms.ESProducer("QGLikelihoodESProducer",
# this is what it makes available
    algo = cms.string('QGL_AK4PF')
)
QGLikelihoodESProducer_AK4PFchs = cms.ESProducer("QGLikelihoodESProducer",
# this is what it makes available
    algo = cms.string('QGL_AK4PFchs')
)
