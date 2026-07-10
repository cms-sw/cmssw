import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer

ScoutingPi0Monitor = DQMEDAnalyzer("ScoutingPi0Analyzer",
                                   scoutingCollection = cms.InputTag('hltScoutingPFPacker'),
                                   minPt = cms.double(1.5),
                                   maxEta = cms.double(2.5),
                                   isolationCone = cms.double(0.2),
                                   isolationPtRatio = cms.double(0.8),
                                   pairMaxDr = cms.double(0.1),
                                   asymmetryCut = cms.double(0.85),
                                   pairMinPt = cms.double(2),
                                   maxMass = cms.double(1.))

