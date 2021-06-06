import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

#import FWCore.Framework.test.cmsExceptionsFatal_cff
#process.options = FWCore.Framework.test.cmsExceptionsFatal_cff.options

process.load("FWCore.MessageService.test.Services_cff")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    debugModules = cms.untracked.vstring('*'),
    files = cms.untracked.PSet(
        sig_debugs = cms.untracked.PSet(
            FwkReport = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            noTimeStamps = cms.untracked.bool(True),
            preEventProcessing = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            threshold = cms.untracked.string('DEBUG')
        ),
        sig_infos = cms.untracked.PSet(
            FwkReport = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            noTimeStamps = cms.untracked.bool(True),
            preEventProcessing = cms.untracked.PSet(
                limit = cms.untracked.int32(0)
            ),
            threshold = cms.untracked.string('INFO')
        )
    )
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(9)
)

process.source = cms.Source("EmptySource")
# process.source = cms.Source("EmptySource",
#    experimentType = cms.untracked.string("Unknown") )


process.sendSomeMessages = cms.EDAnalyzer("makeSignals")

process.p = cms.Path(process.sendSomeMessages)


