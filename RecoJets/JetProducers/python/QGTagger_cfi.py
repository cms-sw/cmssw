import FWCore.ParameterSet.Config as cms

# v0: only for tests (old PDFs retrieved with rho kt6PFJets binning, at 8TeV)
qgDatabaseVersion = 'v0-test'

from CondCore.DBCommon.CondDBSetup_cfi import *
QGPoolDBESSource = cms.ESSource("PoolDBESSource",
      CondDBSetup,
      toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('QGLikelihoodRcd'),
            tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK5PFchs'),
            label  = cms.untracked.string('QGL_AK5PFchs')
        ),
        cms.PSet(
            record = cms.string('QGLikelihoodRcd'),
            tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK5PF'),
            label  = cms.untracked.string('QGL_AK5PF')
        ),
        cms.PSet(
            record = cms.string('QGLikelihoodSystematicsRcd'),
            tag    = cms.string('QGLikelihoodSystematicsObject_'+qgDatabaseVersion+'_Pythia'),
            label  = cms.untracked.string('QGL_Syst_Pythia')
        ),
        cms.PSet(
            record = cms.string('QGLikelihoodSystematicsRcd'),
            tag    = cms.string('QGLikelihoodSystematicsObject_'+qgDatabaseVersion+'_Herwig++'),
            label  = cms.untracked.string('QGL_Syst_Herwig++')
        ),
      ),
      connect = cms.string('frontier://FrontierProd/CMS_COND_PAT_000'),
)

QGTagger = cms.EDProducer('QGTagger',
  srcRho 		= cms.InputTag('fixedGridRhoFastjetAll'),		
  srcVertexCollection	= cms.InputTag('offlinePrimaryVerticesWithBS'),
  jec			= cms.string(''),
  systematicsLabel	= cms.string('')
)
