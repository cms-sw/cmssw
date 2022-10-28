import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.RecoB.bTagMiniDQMDeepFlavour import *
from DQMOffline.RecoB.bTagMiniDQMDeepCSV import *

from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import patJets



# add jets with pfSecondaryVertexTagInfos
patJetsSVInfo = patJets.clone(
    tagInfoSources = ['pfSecondaryVertexTagInfos'],
    addTagInfos = True
)
patJetsSVInfoTask = cms.Task(patJetsSVInfo)


bTagSVDQM = DQMEDAnalyzer('MiniAODSVAnalyzer',
                          cms.PSet(JetTag = cms.InputTag('patJetsSVInfo'),
                                   svTagInfo = cms.string('pfSecondaryVertex'),
                                   ptMin = cms.double(30.),
                                   etaMax = cms.double(2.5),
                                   )
                          )


bTagMiniDQMGlobal = cms.PSet(
    JetTag = cms.InputTag('slimmedJets'),
    MClevel = cms.int32(0),
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


def addSequences(Analyzer, Harvester, discriminators, regions, globalPSet, label='bTag'):
    for discr in discriminators.keys():
        for region in regions.keys():
            name = label + discr + region

            globals()[name + 'Analyzer'] = DQMEDAnalyzer('MiniAODTaggerAnalyzer',    cms.PSet(globalPSet, discriminators[discr], regions[region]))
            globals()[name + 'Harvester'] = DQMEDHarvester('MiniAODTaggerHarvester', cms.PSet(globalPSet, discriminators[discr], regions[region]))

            Analyzer.insert(-1, globals()[name + 'Analyzer'])
            Harvester.insert(-1, globals()[name + 'Harvester'])



bTagMiniDQMSource = cms.Sequence(bTagSVDQM, patJetsSVInfoTask)
bTagMiniDQMHarvesting = cms.Sequence()

addSequences(bTagMiniDQMSource,
             bTagMiniDQMHarvesting,
             discriminators=DeepFlavourDiscriminators,
             regions=Etaregions,
             globalPSet=bTagMiniDQMGlobal,
             label='bTagDeepFlavourDQM')

addSequences(bTagMiniDQMSource,
             bTagMiniDQMHarvesting,
             discriminators=DeepCSVDiscriminators,
             regions=Etaregions,
             globalPSet=bTagMiniDQMGlobal,
             label='bTagDeepCSVDQM')



# Validation addSequences

bTagMiniValidationGlobal = bTagMiniDQMGlobal.clone(
    MClevel = 1 # produce flavour plots for b, c ,light (dusg)
)

bTagMiniValidationSource = cms.Sequence(bTagSVDQM, patJetsSVInfoTask)
bTagMiniValidationHarvesting = cms.Sequence()


addSequences(bTagMiniValidationSource,
             bTagMiniValidationHarvesting,
             discriminators=DeepFlavourDiscriminators,
             regions={'Global': Etaregions['Global']}, # only for global Eta range
             globalPSet=bTagMiniValidationGlobal,
             label='bTagDeepFlavourValidation')

addSequences(bTagMiniValidationSource,
             bTagMiniValidationHarvesting,
             discriminators=DeepCSVDiscriminators,
             regions={'Global': Etaregions['Global']}, # only for global Eta range
             globalPSet=bTagMiniValidationGlobal,
             label='bTagDeepCSVValidation')



from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.ProcessModifiers.miniAOD_skip_trackExtras_cff import miniAOD_skip_trackExtras
from Configuration.Eras.Modifier_run2_miniAOD_94XFall17_cff import run2_miniAOD_94XFall17

_mAOD = (pp_on_AA | miniAOD_skip_trackExtras | run2_miniAOD_94XFall17)
_mAOD.toReplaceWith(bTagMiniDQMSource, bTagMiniDQMSource.copyAndExclude([bTagSVDQM, patJetsSVInfoTask]))
_mAOD.toReplaceWith(bTagMiniValidationSource, bTagMiniValidationSource.copyAndExclude([bTagSVDQM, patJetsSVInfoTask]))
