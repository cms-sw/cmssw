import FWCore.ParameterSet.Config as cms

# See https://twiki.cern.ch/twiki/bin/viewauth/CMS/QGDataBaseVersion
qgDatabaseVersion = 'v1'

from CondCore.DBCommon.CondDBSetup_cfi import *
QGPoolDBESSource = cms.ESSource("PoolDBESSource",
      CondDBSetup,
      toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('QGLikelihoodRcd'),
            tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK4PFchs'),
            label  = cms.untracked.string('QGL_AK4PFchs')
        ),
        cms.PSet(
            record = cms.string('QGLikelihoodRcd'),
            tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK4PF'),
            label  = cms.untracked.string('QGL_AK4PF')
        ),
      ),
      connect = cms.string('frontier://FrontierProd/CMS_COND_PAT_000'),
)

QGTagger = cms.EDProducer('QGTagger',
  srcRho 		= cms.InputTag('fixedGridRhoFastjetAll'),		
  srcVertexCollection	= cms.InputTag('offlinePrimaryVerticesWithBS'),
  useQualityCuts	= cms.bool(False)
)

QGTaggerMiniAOD = cms.EDProducer('QGTagger',
  srcRho 		= cms.InputTag('fixedGridRhoFastjetAll'),		
  srcVertexCollection	= cms.InputTag(''),
  useQualityCuts	= cms.bool(False)
)
