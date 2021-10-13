import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.RecoB.bTagMiniDQMDeepFlavour import *
from DQMOffline.RecoB.bTagMiniDQMDeepCSV import *


bTagMiniDQMGlobal = cms.PSet(
    JetTag = cms.InputTag('slimmedJets'),
    MClevel = cms.int32(4),
    differentialPlots = cms.bool(True),

    ptActive = cms.bool(False),
    ptMin = cms.double(30.),
    ptMax = cms.double(40000.),
)


# Eta regions
Etaregions = {
    'Global': cms.PSet(
        etaActive = cms.bool(False),
        etaMin = cms.double(0.),
        etaMax = cms.double(2.5),
    ),

    'Barrel': cms.PSet(
        etaActive = cms.bool(True),
        etaMin = cms.double(0.),
        etaMax = cms.double(1.4),
    ),

    'Endcap': cms.PSet(
        etaActive = cms.bool(True),
        etaMin = cms.double(1.4),
        etaMax = cms.double(2.5),
    ),
}


def getSequences(discriminators, regions, globalPSet, label='bTag'):
    Analyzer = cms.Sequence()
    Harvester = cms.Sequence()

    for discr in discriminators.keys():
        for region in regions.keys():
            name = label + discr + region

            globals()[name + 'Analyzer'] = DQMEDAnalyzer('MiniAODTaggerAnalyzer',    cms.PSet(globalPSet, discriminators[discr], regions[region]))
            globals()[name + 'Harvester'] = DQMEDHarvester('MiniAODTaggerHarvester', cms.PSet(globalPSet, discriminators[discr], regions[region]))

            Analyzer.insert(-1, globals()[name + 'Analyzer'])
            Harvester.insert(-1, globals()[name + 'Harvester'])

    return Analyzer, Harvester




DeepFlavourAnalyzer, DeepFlavourHarvester = getSequences(discriminators=DeepFlavourDiscriminators,
                                                         regions=Etaregions,
                                                         globalPSet=bTagMiniDQMGlobal,
                                                         label='bTagDeepFlavourDQM')

DeepCSVAnalyzer, DeepCSVHarvester         = getSequences(discriminators=DeepCSVDiscriminators,
                                                         regions=Etaregions,
                                                         globalPSet=bTagMiniDQMGlobal,
                                                         label='bTagDeepCSVDQM')

bTagMiniDQMSource = cms.Sequence(
    DeepFlavourAnalyzer
    #* DeepCSVAnalyzer
)

bTagMiniDQMHarvesting = cms.Sequence(
    DeepFlavourHarvester
    #* DeepCSVHarvester
)
