import FWCore.ParameterSet.Config as cms

igprof = cms.EDAnalyzer("IgProfModule",
  reportEventInterval     = cms.untracked.int32(0),
  reportToFileAtBeginJob  = cms.untracked.string(''), #"|gzip -c>igprof.begin-job.gz"
  reportToFileAtEndJob    = cms.untracked.string(''), #"|gzip -c>igprof.end-job.gz"
  reportToFileAtBeginLumi = cms.untracked.string(''), #"|gzip -c>igprof.%I.%E.%L.%R.begin-lumi.gz"
  reportToFileAtEndLumi   = cms.untracked.string(''), #"|gzip -c>igprof.%I.%E.%L.%R.end-lumi.gz"
  reportToFileAtInputFile = cms.untracked.string(''), #"|gzip -c>igprof.%I.%F.file.gz"
  reportToFileAtEvent     = cms.untracked.string('')) #"|gzip -c>igprof.%I.%E.%L.%R.event.gz"
