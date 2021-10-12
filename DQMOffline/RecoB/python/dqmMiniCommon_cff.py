import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester


bTagMiniAnalyzerGlobal = cms.PSet(
    JetTag = cms.InputTag('slimmedJets'),
    MCplots = cms.bool(True),
    differentialPlots = cms.bool(True),

    ptActive = cms.bool(False),
    ptMin = cms.double(30.),
    ptMax = cms.double(40000.),
)

globalEta = cms.PSet(
    etaActive = cms.bool(False),
    etaMin = cms.double(0.),
    etaMax = cms.double(2.5),
)

barrelEta = cms.PSet(
    etaActive = cms.bool(True),
    etaMin = cms.double(0.),
    etaMax = cms.double(1.4),
)

endcapEta = cms.PSet(
    etaActive = cms.bool(True),
    etaMin = cms.double(1.4),
    etaMax = cms.double(2.5),
)


def getAnalyzerHarvester(parameters):
    return DQMEDAnalyzer('MiniAODTaggerAnalyzer', parameters), DQMEDHarvester('MiniAODTaggerHarvester',parameters)
