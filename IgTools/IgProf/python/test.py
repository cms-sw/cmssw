import FWCore.ParameterSet.Config as cms
from IgTools.IgProf.IgProfTrigger import igprof

process = cms.Process("IGPROF")
process.load("IgTools.IgProf.IgProfTrigger")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1000))
process.p = cms.Path(igprof)

process.igprof.reportEventInterval     = cms.untracked.int32(250)
process.igprof.reportToFileAtBeginJob  = cms.untracked.string("|gzip -c>igprof.begin-job.gz")
#process.igprof.reportToFileAtEndJob    = cms.untracked.string("|gzip -c>igprof.end-job.gz")
#process.igprof.reportToFileAtBeginLumi = cms.untracked.string("|gzip -c>igprof.%I.%E.%L.%R.begin-lumi.gz")
#process.igprof.reportToFileAtEndLumi   = cms.untracked.string("|gzip -c>igprof.%I.%E.%L.%R.end-lumi.gz")
#process.igprof.reportToFileAtInputFile = cms.untracked.string("|gzip -c>igprof.%I.%F.file.gz")
process.igprof.reportToFileAtEvent     = cms.untracked.string("|gzip -c>igprof.%I.%E.%L.%R.event.gz")
