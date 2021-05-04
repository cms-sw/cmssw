import FWCore.ParameterSet.Config as cms

process = cms.Process("HLT")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 999999999
process.MessageLogger.HLTrigReport=dict()

process.options.wantSummary = False

process.source = cms.Source("EmptySource",
                            numberEventsInLuminosityBlock = cms.untracked.uint32(2),
                            numberEventsInRun = cms.untracked.uint32(4))

process.maxEvents.input = 10

process.f1 = cms.EDFilter("Prescaler", 
                          prescaleFactor = cms.int32(3), 
                          prescaleOffset = cms.int32(0) )

process.f2 = cms.EDFilter("Prescaler",
                          prescaleFactor = cms.int32(5), 
                          prescaleOffset = cms.int32(2) )

process.f3 = cms.EDFilter("Prescaler",
                          prescaleFactor = cms.int32(2), 
                          prescaleOffset = cms.int32(1) )

process.pathFilter =cms.EDFilter("PathStatusFilter", logicalExpression = cms.string("p1 or p2"))
process.HLTriggerFinalPath = cms.Path(process.pathFilter)

process.p1 = cms.Path(process.f1)
process.p2 = cms.Path(process.f2+process.f3)

process.load( "HLTrigger.HLTanalyzers.hlTrigReport_cfi" )
process.hlTrigReport.HLTriggerResults   = cms.InputTag("TriggerResults", "", "HLT")
process.hlTrigReport.ReferencePath      = cms.untracked.string( "HLTriggerFinalPath" )
process.hlTrigReport.ReferenceRate      = cms.untracked.double( 100.0 )
process.hlTrigReport.reportBy        = "lumi"

process.report = cms.EndPath( process.hlTrigReport )


