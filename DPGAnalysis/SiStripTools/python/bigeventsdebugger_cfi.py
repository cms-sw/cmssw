import FWCore.ParameterSet.Config as cms

bigeventsdebugger = cms.EDAnalyzer('BigEventsDebugger',
                                   singleEvents = cms.bool(True),
                                   maskedModules = cms.untracked.vuint32()
)
