import FWCore.ParameterSet.Config as cms

anaMET = cms.EDAnalyzer('METAnalyzer',
                        METSrc = cms.InputTag('met'),
)
