import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.bTagGenericAnalysis_cff import bTagGenericAnalysisBlock
from DQMOffline.RecoB.cTagGenericAnalysis_cff import cTagGenericAnalysisBlock


# recommendation for UL18: https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
deepFlavourWP = {
    'BvsAll': 0.0490, # loose
    'CvsL':   0.099,  # medium
    'CvsB':   0.325,  # medium
}


DeepFlavourDiscriminators = {
    'BvsAll': cms.PSet(
        bTagGenericAnalysisBlock,

        folder = cms.string('DeepFlavour_BvsAll'),
        CTagPlots = cms.bool(False),
        discrCut = cms.double(deepFlavourWP['BvsAll']),
        numerator = cms.vstring(
            'pfDeepFlavourJetTags:probb',
            'pfDeepFlavourJetTags:probbb',
            'pfDeepFlavourJetTags:problepb',
        ),
        denominator = cms.vstring(),
    ),

    'CvsL': cms.PSet(
        cTagGenericAnalysisBlock,

        folder = cms.string('DeepFlavour_CvsL'),
        CTagPlots = cms.bool(True),
        discrCut = cms.double(deepFlavourWP['CvsL']),
        numerator = cms.vstring('pfDeepFlavourJetTags:probc'),
        denominator = cms.vstring(
            'pfDeepFlavourJetTags:probc',
            'pfDeepFlavourJetTags:probuds',
            'pfDeepFlavourJetTags:probg',
        ),
    ),

    'CvsB': cms.PSet(
        cTagGenericAnalysisBlock,

        folder = cms.string('DeepFlavour_CvsB'),
        CTagPlots = cms.bool(True),
        discrCut = cms.double(deepFlavourWP['CvsB']),
        numerator = cms.vstring('pfDeepFlavourJetTags:probc'),
        denominator = cms.vstring(
            'pfDeepFlavourJetTags:probc',
            'pfDeepFlavourJetTags:probb',
            'pfDeepFlavourJetTags:probbb',
            'pfDeepFlavourJetTags:problepb',
        ),
    ),
}
