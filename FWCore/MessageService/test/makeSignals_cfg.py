import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

#import FWCore.Framework.test.cmsExceptionsFatal_cff
#process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    sig_debugs = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG'),
        noTimeStamps = cms.untracked.bool(True),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    debugModules = cms.untracked.vstring('*'),
    sig_infos = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        noTimeStamps = cms.untracked.bool(True),
        FwkReport = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        preEventProcessing = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        )
    ),
    categories = cms.untracked.vstring('preEventProcessing', 
        'FwkReport'),
    destinations = cms.untracked.vstring('sig_infos', 
        'sig_debugs')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(9)
)

process.source = cms.Source("EmptySource")
# process.source = cms.Source("EmptySource",
#    experimentType = cms.untracked.string("Unknown") )


process.sendSomeMessages = cms.EDAnalyzer("makeSignals")

process.p = cms.Path(process.sendSomeMessages)


