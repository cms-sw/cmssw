import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.dqmMiniCommon_cff import bTagMiniAnalyzerGlobal, globalEta, barrelEta, endcapEta, getAnalyzerHarvester
from DQMOffline.RecoB.bTagGenericAnalysis_cff import bTagGenericAnalysisBlock
from DQMOffline.RecoB.cTagGenericAnalysis_cff import cTagGenericAnalysisBlock


# recommendation for UL18: https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL18
deepFlavourWP = {
    'BvsAll': 0.0490, # loose
    'CvsL':   0.099,  # medium
    'CvsB':   0.325,  # medium
}

BvsAll = cms.PSet(
    bTagGenericAnalysisBlock,
    bTagMiniAnalyzerGlobal,
    folder = cms.string('deepFlavour_BvsAll'),
    CTagPlots = cms.bool(False),
    discrCut = cms.double(deepFlavourWP['BvsAll']),
    numerator = cms.vstring(
        'pfDeepFlavourJetTags:probb',
        'pfDeepFlavourJetTags:probbb',
        'pfDeepFlavourJetTags:problepb',
    ),
    denominator = cms.vstring(),
)

CvsL = cms.PSet(
    cTagGenericAnalysisBlock,
    bTagMiniAnalyzerGlobal,
    folder = cms.string('deepFlavour_CvsL'),
    CTagPlots = cms.bool(True),
    discrCut = cms.double(deepFlavourWP['CvsL']),
    numerator = cms.vstring('pfDeepFlavourJetTags:probc'),
    denominator = cms.vstring(
        'pfDeepFlavourJetTags:probc',
        'pfDeepFlavourJetTags:probuds',
        'pfDeepFlavourJetTags:probg',
    ),
)

CvsB = cms.PSet(
    cTagGenericAnalysisBlock,
    bTagMiniAnalyzerGlobal,
    folder = cms.string('deepFlavour_CvsB'),
    CTagPlots = cms.bool(True),
    discrCut = cms.double(deepFlavourWP['CvsB']),
    numerator = cms.vstring('pfDeepFlavourJetTags:probc'),
    denominator = cms.vstring(
        'pfDeepFlavourJetTags:probc',
        'pfDeepFlavourJetTags:probb',
        'pfDeepFlavourJetTags:probbb',
        'pfDeepFlavourJetTags:problepb',
    ),
)



# DeepFlavourBvsAll
bTagDeepFlavourBvsAllAnalyzer,       bTagDeepFlavourBvsAllHarvester       = getAnalyzerHarvester(cms.PSet(globalEta, BvsAll))
bTagDeepFlavourBvsAllBarrelAnalyzer, bTagDeepFlavourBvsAllBarrelHarvester = getAnalyzerHarvester(cms.PSet(barrelEta, BvsAll))
bTagDeepFlavourBvsAllEndcapAnalyzer, bTagDeepFlavourBvsAllEndcapHarvester = getAnalyzerHarvester(cms.PSet(endcapEta, BvsAll))


# DeepFlavourCvsL
bTagDeepFlavourCvsLAnalyzer,       bTagDeepFlavourCvsLHarvester       = getAnalyzerHarvester(cms.PSet(globalEta, CvsL))
bTagDeepFlavourCvsLBarrelAnalyzer, bTagDeepFlavourCvsLBarrelHarvester = getAnalyzerHarvester(cms.PSet(barrelEta, CvsL))
bTagDeepFlavourCvsLEndcapAnalyzer, bTagDeepFlavourCvsLEndcapHarvester = getAnalyzerHarvester(cms.PSet(endcapEta, CvsL))


# DeepFlavourCvsB
bTagDeepFlavourCvsBAnalyzer,       bTagDeepFlavourCvsBHarvester       = getAnalyzerHarvester(cms.PSet(globalEta ,CvsB))
bTagDeepFlavourCvsBBarrelAnalyzer, bTagDeepFlavourCvsBBarrelHarvester = getAnalyzerHarvester(cms.PSet(barrelEta, CvsB))
bTagDeepFlavourCvsBEndcapAnalyzer, bTagDeepFlavourCvsBEndcapHarvester = getAnalyzerHarvester(cms.PSet(endcapEta, CvsB))



DeepFlavourAnalyzer = cms.Sequence(
    bTagDeepFlavourBvsAllAnalyzer *
    bTagDeepFlavourBvsAllBarrelAnalyzer *
    bTagDeepFlavourBvsAllEndcapAnalyzer *

    bTagDeepFlavourCvsLAnalyzer *
    bTagDeepFlavourCvsLBarrelAnalyzer *
    bTagDeepFlavourCvsLEndcapAnalyzer *

    bTagDeepFlavourCvsBAnalyzer *
    bTagDeepFlavourCvsBBarrelAnalyzer *
    bTagDeepFlavourCvsBEndcapAnalyzer
)



DeepFlavourHarvester = cms.Sequence(
    bTagDeepFlavourBvsAllHarvester *
    bTagDeepFlavourBvsAllBarrelHarvester *
    bTagDeepFlavourBvsAllEndcapHarvester *

    bTagDeepFlavourCvsLHarvester *
    bTagDeepFlavourCvsLBarrelHarvester *
    bTagDeepFlavourCvsLEndcapHarvester *

    bTagDeepFlavourCvsBHarvester *
    bTagDeepFlavourCvsBBarrelHarvester *
    bTagDeepFlavourCvsBEndcapHarvester
)
