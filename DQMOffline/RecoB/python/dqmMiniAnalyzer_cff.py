import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

from DQMOffline.RecoB.bTagGenericAnalysis_cff import bTagGenericAnalysisBlock


bTagAnalyzer_mini = cms.Sequence()
bTagHarvester_mini = cms.Sequence()

def addToSequences(label, parameters):
    globals()[label + 'Analyzer'] = DQMEDAnalyzer('MiniAODTaggerAnalyzer', parameters)
    globals()[label + 'Harvester'] = DQMEDHarvester('MiniAODTaggerHarvester', parameters)

    bTagAnalyzer_mini.insert(-1, globals()[label + 'Analyzer'])
    bTagHarvester_mini.insert(-1, globals()[label + 'Harvester'])



bTagMiniAnalyzerGlobal = cms.PSet(
    JetTag = cms.InputTag('slimmedJets'),
    differentialPlots = cms.bool(True),

    etaActive = cms.bool(False),
    etaMin = cms.double(0.),
    etaMax = cms.double(2.5),
    ptActive = cms.bool(False),
    ptMin = cms.double(0.),
    ptMax = cms.double(0.),
)



addToSequences('bTagDeepFlavourBvsAll',
    cms.PSet(
        bTagGenericAnalysisBlock,
        bTagMiniAnalyzerGlobal,

        folder = cms.string('deepFlavour_BvsAll'),
        CTagPlots = cms.bool(False),
        discrCut = cms.double(-999),
        numerator = cms.vstring(
            'pfDeepFlavourJetTags:probb',
            'pfDeepFlavourJetTags:probbb',
            'pfDeepFlavourJetTags:problepb',
        ),
        denominator = cms.vstring(),
    )
)


addToSequences('bTagDeepFlavourCvsB',
    cms.PSet(
        bTagGenericAnalysisBlock,
        bTagMiniAnalyzerGlobal,

        folder = cms.string('deepFlavour_CvsB'),
        CTagPlots = cms.bool(True),
        discrCut = cms.double(-999),
        numerator = cms.vstring('pfDeepFlavourJetTags:probc'),
        denominator = cms.vstring(
            'pfDeepFlavourJetTags:probc',
            'pfDeepFlavourJetTags:probb',
            'pfDeepFlavourJetTags:probbb',
            'pfDeepFlavourJetTags:problepb',
        ),
    )
)


addToSequences('bTagDeepFlavourCvsL',
    cms.PSet(
        bTagGenericAnalysisBlock,
        bTagMiniAnalyzerGlobal,

        folder = cms.string('deepFlavour_CvsL'),
        CTagPlots = cms.bool(True),
        discrCut = cms.double(-999),
        numerator = cms.vstring('pfDeepFlavourJetTags:probc'),
        denominator = cms.vstring(
            'pfDeepFlavourJetTags:probc',
            'pfDeepFlavourJetTags:probuds',
            'pfDeepFlavourJetTags:probg',
        ),
    )
)
