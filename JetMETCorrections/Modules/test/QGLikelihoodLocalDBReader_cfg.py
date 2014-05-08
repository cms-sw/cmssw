import FWCore.ParameterSet.Config as cms

process = cms.Process("LikelihoodDBLocalReader")

process.MessageLogger = cms.Service("MessageLogger",
            destinations = cms.untracked.vstring('cout'),
            cout = cms.untracked.PSet(threshold = cms.untracked.string( 'INFO' )),
)

process.load('Configuration.StandardSequences.Services_cff')
process.load("JetMETCorrections.Modules.qglESProducer_cfi")

from CondCore.DBCommon.CondDBSetup_cfi import *

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.source = cms.Source("EmptySource")

qgDatabaseVersion = 'v0-test'
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
      CondDBSetup,
      toGet = cms.VPSet(
        cms.PSet(
            record = cms.string('QGLikelihoodRcd'),
            tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK5PF'),
            label  = cms.untracked.string('QGL_AK5PF')
        ),
        cms.PSet(
            record = cms.string('QGLikelihoodRcd'),
            tag    = cms.string('QGLikelihoodObject_'+qgDatabaseVersion+'_AK5PFchs'),
            label  = cms.untracked.string('QGL_AK5PFchs')
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
      connect = cms.string('sqlite:QGL_'+qgDatabaseVersion+'.db')
)


process.demo1 = cms.EDAnalyzer('QGLikelihoodDBReader', 
        payloadName    = cms.untracked.string('QGL_AK5PF'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True)
)

process.demo2 = cms.EDAnalyzer('QGLikelihoodDBReader', 
        payloadName    = cms.untracked.string('QGL_AK5PFchs'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True)
)

process.demo3 = cms.EDAnalyzer('QGLikelihoodSystematicsDBReader', 
        payloadName    = cms.untracked.string('QGL_Syst_Pythia'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True)
)

process.demo4 = cms.EDAnalyzer('QGLikelihoodSystematicsDBReader', 
        payloadName    = cms.untracked.string('QGL_Syst_Herwig++'),
        printScreen    = cms.untracked.bool(False),
        createTextFile = cms.untracked.bool(True)
)

process.p = cms.Path(process.demo1 * process.demo2 * process.demo3 * process.demo4)
