import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Modules.qglESProducer_cfi import *
from CondCore.DBCommon.CondDBSetup_cfi import *


PoolDBESSource = cms.ESSource("PoolDBESSource",
      CondDBSetup,
      toGet = cms.VPSet(
      cms.PSet(
            record = cms.string('QGLikelihoodRcd'),
            tag    = cms.string('QGLikelihoodObject_V1_AK5PF'),
            label  = cms.untracked.string('QGL_AK5PF')
            ),
      cms.PSet(
            record = cms.string('QGLikelihoodRcd'),
            tag    = cms.string('QGLikelihoodObject_V1_AK5PFchs'),
            label  = cms.untracked.string('QGL_AK5PFchs')
            ),
        ),
      connect = cms.string('sqlite:QGL_V1.db')
)

QGTagger = cms.EDProducer('QGTagger',
  srcRho 		= cms.InputTag('fixedGridRhoFastjetAll'),
  srcVertexCollection	= cms.InputTag('offlinePrimaryVerticesWithBS'),
  QGLParameters 	= cms.string('QGL_AK5PFchs'),
  jec			= cms.string(''),
)
