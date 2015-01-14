import FWCore.ParameterSet.Config as cms

# See https://twiki.cern.ch/twiki/bin/viewauth/CMS/QGDataBaseVersion
qgDatabaseVersion = 'v1'

from CondCore.DBCommon.CondDBSetup_cfi import *
QGPoolDBESSource = cms.ESSource("PoolDBESSource",
      CondDBSetup,
      toGet = cms.VPSet(),
      connect = cms.string('frontier://FrontierProd/CMS_COND_PAT_000'),
)

for type in ['AK4PF','AK4PFchs','AK4PF_antib','AK4PFchs_antib']:
  QGPoolDBESSource.toGet.extend(cms.VPSet(cms.PSet(
    record = cms.string('QGLikelihoodRcd'),
    tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_'+type),
    label  = cms.untracked.string('QGL_'+type)
  )))

QGTagger = cms.EDProducer('QGTagger',
  srcJets		= cms.InputTag('ak4PFJetsCHS'),
  jetsLabel		= cms.string('QGL_AK4PFchs'),
  srcRho 		= cms.InputTag('fixedGridRhoFastjetAll'),		
  srcVertexCollection	= cms.InputTag('offlinePrimaryVerticesWithBS'),
  useQualityCuts	= cms.bool(False)
)
