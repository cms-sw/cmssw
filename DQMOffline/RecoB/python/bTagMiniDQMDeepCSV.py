import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.bTagGenericAnalysis_cff import bTagGenericAnalysisBlock
from DQMOffline.RecoB.cTagGenericAnalysis_cff import cTagGenericAnalysisBlock


# recommendation for UL18: https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
deepCSVWP = {
    'BvsAll': 0.1208, # loose
    'CvsL':   0.153,  # medium
    'CvsB':   0.363,  # medium
}


DeepCSVDiscriminators = {
    'BvsAll': cms.PSet(
        bTagGenericAnalysisBlock,

        folder = cms.string('DeepCSV_BvsAll'),
        CTagPlots = cms.bool(False),
        discrCut = cms.double(deepCSVWP['BvsAll']),
        numerator = cms.vstring(
            'pfDeepCSVJetTags:probb',
            'pfDeepCSVJetTags:probbb',
        ),
        denominator = cms.vstring(),
    ),

    'CvsL': cms.PSet(
        cTagGenericAnalysisBlock,

        folder = cms.string('DeepCSV_CvsL'),
        CTagPlots = cms.bool(True),
        discrCut = cms.double(deepCSVWP['CvsL']),
        numerator = cms.vstring('pfDeepCSVJetTags:probc'),
        denominator = cms.vstring(
            'pfDeepCSVJetTags:probc',
            'pfDeepCSVJetTags:probudsg',
        ),
    ),

    'CvsB': cms.PSet(
        cTagGenericAnalysisBlock,

        folder = cms.string('DeepCSV_CvsB'),
        CTagPlots = cms.bool(True),
        discrCut = cms.double(deepCSVWP['CvsB']),
        numerator = cms.vstring('pfDeepCSVJetTags:probc'),
        denominator = cms.vstring(
            'pfDeepCSVJetTags:probc',
            'pfDeepCSVJetTags:probb',
            'pfDeepCSVJetTags:probbb',
        ),
    ),
}
