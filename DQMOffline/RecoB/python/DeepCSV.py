import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.dqmMiniCommon_cff import bTagMiniAnalyzerGlobal, globalEta, barrelEta, endcapEta, getAnalyzerHarvester
from DQMOffline.RecoB.bTagGenericAnalysis_cff import bTagGenericAnalysisBlock
from DQMOffline.RecoB.cTagGenericAnalysis_cff import cTagGenericAnalysisBlock


# recommendation for UL18: https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
deepCSVWP = {
    'BvsAll': 0.1208, # loose
    'CvsL':   0.153,  # medium
    'CvsB':   0.363,  # medium
}

BvsAll = cms.PSet(
    bTagGenericAnalysisBlock,
    bTagMiniAnalyzerGlobal,
    folder = cms.string('DeepCSV_BvsAll'),
    CTagPlots = cms.bool(False),
    discrCut = cms.double(deepCSVWP['BvsAll']),
    numerator = cms.vstring(
        'pfDeepCSVJetTags:probb',
        'pfDeepCSVJetTags:probbb',
    ),
    denominator = cms.vstring(),
)

CvsL = cms.PSet(
    cTagGenericAnalysisBlock,
    bTagMiniAnalyzerGlobal,
    folder = cms.string('DeepCSV_CvsL'),
    CTagPlots = cms.bool(True),
    discrCut = cms.double(deepCSVWP['CvsL']),
    numerator = cms.vstring('pfDeepCSVJetTags:probc'),
    denominator = cms.vstring(
        'pfDeepCSVJetTags:probc',
        'pfDeepCSVJetTags:probudsg',
    ),
)

CvsB = cms.PSet(
    cTagGenericAnalysisBlock,
    bTagMiniAnalyzerGlobal,
    folder = cms.string('DeepCSV_CvsB'),
    CTagPlots = cms.bool(True),
    discrCut = cms.double(deepCSVWP['CvsB']),
    numerator = cms.vstring('pfDeepCSVJetTags:probc'),
    denominator = cms.vstring(
        'pfDeepCSVJetTags:probc',
        'pfDeepCSVJetTags:probb',
        'pfDeepCSVJetTags:probbb',
    ),
)



# DeepCSVBvsAll
bTagDeepCSVBvsAllAnalyzer,       bTagDeepCSVBvsAllHarvester       = getAnalyzerHarvester(cms.PSet(globalEta, BvsAll))
bTagDeepCSVBvsAllBarrelAnalyzer, bTagDeepCSVBvsAllBarrelHarvester = getAnalyzerHarvester(cms.PSet(barrelEta, BvsAll))
bTagDeepCSVBvsAllEndcapAnalyzer, bTagDeepCSVBvsAllEndcapHarvester = getAnalyzerHarvester(cms.PSet(endcapEta, BvsAll))


# DeepCSVCvsL
bTagDeepCSVCvsLAnalyzer,       bTagDeepCSVCvsLHarvester       = getAnalyzerHarvester(cms.PSet(globalEta, CvsL))
bTagDeepCSVCvsLBarrelAnalyzer, bTagDeepCSVCvsLBarrelHarvester = getAnalyzerHarvester(cms.PSet(barrelEta, CvsL))
bTagDeepCSVCvsLEndcapAnalyzer, bTagDeepCSVCvsLEndcapHarvester = getAnalyzerHarvester(cms.PSet(endcapEta, CvsL))


# DeepCSVCvsB
bTagDeepCSVCvsBAnalyzer,       bTagDeepCSVCvsBHarvester       = getAnalyzerHarvester(cms.PSet(globalEta ,CvsB))
bTagDeepCSVCvsBBarrelAnalyzer, bTagDeepCSVCvsBBarrelHarvester = getAnalyzerHarvester(cms.PSet(barrelEta, CvsB))
bTagDeepCSVCvsBEndcapAnalyzer, bTagDeepCSVCvsBEndcapHarvester = getAnalyzerHarvester(cms.PSet(endcapEta, CvsB))



DeepCSVAnalyzer = cms.Sequence(
    bTagDeepCSVBvsAllAnalyzer *
    bTagDeepCSVBvsAllBarrelAnalyzer *
    bTagDeepCSVBvsAllEndcapAnalyzer *

    bTagDeepCSVCvsLAnalyzer *
    bTagDeepCSVCvsLBarrelAnalyzer *
    bTagDeepCSVCvsLEndcapAnalyzer *

    bTagDeepCSVCvsBAnalyzer *
    bTagDeepCSVCvsBBarrelAnalyzer *
    bTagDeepCSVCvsBEndcapAnalyzer
)



DeepCSVHarvester = cms.Sequence(
    bTagDeepCSVBvsAllHarvester *
    bTagDeepCSVBvsAllBarrelHarvester *
    bTagDeepCSVBvsAllEndcapHarvester *

    bTagDeepCSVCvsLHarvester *
    bTagDeepCSVCvsLBarrelHarvester *
    bTagDeepCSVCvsLEndcapHarvester *

    bTagDeepCSVCvsBHarvester *
    bTagDeepCSVCvsBBarrelHarvester *
    bTagDeepCSVCvsBEndcapHarvester
)
