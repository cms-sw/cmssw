import FWCore.ParameterSet.Config as cms

#Log-Report ---------- HLTLogMonitorFilter Summary ------------
#Log-Report  Threshold     Issued   Accepted   Rejected   Prescale Category
#Log-Report         10        100         19         81        100 Other   
#Log-Report          0        100          0        100          1 Test    
#Log-Report          1        150        150          0          1 TestError
#Log-Report         20       1100         40       1060        400 TestWarning
#
#TrigReport ---------- Event  Summary ------------
#TrigReport Events total = 1000 passed = 135 failed = 865

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 1000

process.maxEvents = cms.untracked.PSet( 
  input = cms.untracked.int32(10000) 
)

#process.options = cms.untracked.PSet(
#  wantSummary = cms.untracked.bool(True)
#)

process.source = cms.Source('EmptySource')

process.error = cms.EDAnalyzer('ArbitraryLogError',
  severity = cms.string('Error'),
  category = cms.string('TestError'),
  rate     = cms.uint32(20)
)

process.warning = cms.EDAnalyzer('ArbitraryLogError',
  severity = cms.string('Warning'),
  category = cms.string('TestWarning'),
  rate     = cms.uint32(1)
)

process.combined = cms.EDAnalyzer('ArbitraryLogError',
  severity = cms.string('Warning'),
  category = cms.string('Test|TestWarning|TestError|Other'),
  rate     = cms.uint32(10)
)

process.reject = cms.EDFilter( "HLTBool",
    result = cms.bool( False )
)

process.hltLogMonitorFilter = cms.EDFilter("HLTLogMonitorFilter",
    default_threshold = cms.uint32(10),
    categories = cms.VPSet(
        cms.PSet(
            name = cms.string('TestWarning'),
            threshold = cms.uint32(20)
        ),
        cms.PSet(
            name = cms.string('TestError'),
            threshold = cms.uint32(1)
        ),
        cms.PSet(
            name = cms.string('Test'),
            threshold = cms.uint32(0)
        )
    )
)

process.path        = cms.Path(process.warning + process.error + process.combined + process.reject)
process.logmonitor  = cms.Path(process.hltLogMonitorFilter)
