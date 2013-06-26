import FWCore.ParameterSet.Config as cms

# Bit Plotting
hltMonBitSummary = cms.EDAnalyzer("HLTMonBitSummary",
                                  directory = cms.untracked.string('myDirectory'),
                                  histLabel = cms.untracked.string('myHistLabel'),
                                  #label = cms.string('myLabel'),
                                  #out = cms.untracked.string('dqm.root'),
                                  HLTPaths = cms.vstring('HLT_.*'),
                                  denominatorWild = cms.untracked.string(''),
                                  denominator = cms.untracked.string(''),
                                  eventSetupPathsKey = cms.untracked.string(''),
                                  TriggerResultsTag = cms.InputTag('TriggerResults','','HLT')
)
