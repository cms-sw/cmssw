import FWCore.ParameterSet.Config as cms
QGLikelihoodESProducer_AK5PF = cms.ESProducer("QGLikelihoodESProducer",
# this is what it makes available
    algo = cms.string('QGL_AK5PF')
)
QGLikelihoodESProducer_AK5PFchs = cms.ESProducer("QGLikelihoodESProducer",
# this is what it makes available
    algo = cms.string('QGL_AK5PFchs')
)
