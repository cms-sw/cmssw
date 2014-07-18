import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
#process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.MessageLogger = cms.Service("MessageLogger",
            destinations = cms.untracked.vstring(
                    'cout'
            ),
            cout = cms.untracked.PSet(
                    threshold = cms.untracked.string( 'INFO' )
            ),
)

process.load('Configuration.StandardSequences.Services_cff')

process.load("JetMETCorrections.Modules.qglESProducer_cfi")

from CondCore.DBCommon.CondDBSetup_cfi import *

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1)
        )

process.source = cms.Source("EmptySource")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
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

process.p = cms.Path(process.demo1 * process.demo2 )
