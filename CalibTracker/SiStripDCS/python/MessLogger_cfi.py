import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service("MessageLogger",
           suppressInfo = cms.untracked.vstring(),
           suppressDebug = cms.untracked.vstring(),
           suppressWarning = cms.untracked.vstring(),

           info = cms.untracked.PSet(
                threshold = cms.untracked.string('INFO'),
                #            limit = cms.untracked.int32(100000),
                noLineBreaks = cms.untracked.bool(False)
                         ),
           debug = cms.untracked.PSet(
                threshold = cms.untracked.string('DEBUG'),
                #            limit = cms.untracked.int32(100000),
                noLineBreaks = cms.untracked.bool(False)
                         ),
           warning = cms.untracked.PSet(
                threshold = cms.untracked.string('WARNING'),
                #            limit = cms.untracked.int32(100000),
                noLineBreaks = cms.untracked.bool(False)
                         ),
           cerr = cms.untracked.PSet(
                threshold = cms.untracked.string('ERROR'),
                #            limit = cms.untracked.int32(100000),
                noLineBreaks = cms.untracked.bool(False)
                         ),
           error = cms.untracked.PSet(
                threshold = cms.untracked.string('ERROR'),
                #            limit = cms.untracked.int32(100000),
                noLineBreaks = cms.untracked.bool(False)
                         ),

           debugModules = cms.untracked.vstring('*'),
           destinations = cms.untracked.vstring('cerr',
                                                'info',
                                                'warning',
                                                'debug',
                                                'error')
)
