import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.tagGenericAnalysis_cff import bTagGenericAnalysisBlock
from DQMOffline.RecoB.tagGenericAnalysis_cff import cTagGenericAnalysisBlock
from DQMOffline.RecoB.tagGenericAnalysis_cff import tauTagGenericAnalysisBlock
from DQMOffline.RecoB.tagGenericAnalysis_cff import sTagGenericAnalysisBlock
from DQMOffline.RecoB.tagGenericAnalysis_cff import qgTagGenericAnalysisBlock

############################################################
#
# DeepCSV
#
############################################################
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


############################################################
#
# DeepFlavour (aka DeepJet)
#
############################################################
# Summer23BPix Working points
deepFlavourWP = {
    'BvsAll': 0.048, # loose
    'CvsL':   0.102,  # medium
    'CvsB':   0.328,  # medium
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

############################################################
#
# AK4 ParticleNet for Puppi jets
#
############################################################
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiCentralJetTagsMetaDiscr as pfParticleNetFromMiniAODAK4PuppiCentralJetTagsMetaDiscr
from RecoBTag.ONNXRuntime.pfParticleNetFromMiniAODAK4_cff import _pfParticleNetFromMiniAODAK4PuppiForwardJetTagsMetaDiscr as pfParticleNetFromMiniAODAK4PuppiForwardJetTagsMetaDiscr

ParticleNetPuppiCentralDiscriminators = {}

for meta_tagger in pfParticleNetFromMiniAODAK4PuppiCentralJetTagsMetaDiscr:
    discr = meta_tagger.split(':')[1]

    commonTaggerConfig = cms.PSet(
        folder = cms.string('ParticleNetCentral_'+discr),
        numerator = cms.vstring(meta_tagger),
        denominator = cms.vstring(),
        discrCut = cms.double(0.3),#Dummy,
        CTagPlots = cms.bool(False)
    )
    if "Bvs" in discr:
        ParticleNetPuppiCentralDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            bTagGenericAnalysisBlock
        )
        if "BvsAll" in discr:
            ParticleNetPuppiCentralDiscriminators[discr].discrCut = cms.double(0.0359)#Summer23BPix Loose WP
    elif "Cvs" in discr:
        ParticleNetPuppiCentralDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            cTagGenericAnalysisBlock,
        )
        ParticleNetPuppiCentralDiscriminators[discr].CTagPlots = True
        if "CvsB" in discr:
            ParticleNetPuppiCentralDiscriminators[discr].discrCut = cms.double(0.358)#Summer23BPix Medium WP
        if "CvsL" in discr:
            ParticleNetPuppiCentralDiscriminators[discr].discrCut = cms.double(0.149)#Summer23BPix Medium WP
    elif "TauVs" in discr:
        ParticleNetPuppiCentralDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            cTagGenericAnalysisBlock
        )
    elif "QvsG" in discr:
        ParticleNetPuppiCentralDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            cTagGenericAnalysisBlock
        )

ParticleNetPuppiForwardDiscriminators = {}

for meta_tagger in pfParticleNetFromMiniAODAK4PuppiForwardJetTagsMetaDiscr:
    discr = meta_tagger.split(':')[1]

    commonTaggerConfig = cms.PSet(
        folder = cms.string('ParticleNetForward_'+discr),
        numerator = cms.vstring(meta_tagger),
        denominator = cms.vstring(),
        discrCut = cms.double(0.3),#Dummy,
        CTagPlots = cms.bool(False)
    )
    if "QvsG" in discr:
        ParticleNetPuppiForwardDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            qgTagGenericAnalysisBlock,
        )

############################################################
#
# UParT
#
############################################################
from RecoBTag.ONNXRuntime.pfUnifiedParticleTransformerAK4_cff import _pfUnifiedParticleTransformerAK4JetTagsMetaDiscrs as pfUnifiedParticleTransformerAK4JetTagsMetaDiscrs

UParTDiscriminators = {}
#
#
#
for meta_tagger in pfUnifiedParticleTransformerAK4JetTagsMetaDiscrs:
    discr = meta_tagger.split(':')[1] #split input tag to get thcde producer label
    #
    #
    #
    commonTaggerConfig = cms.PSet(
        folder = cms.string('UParT_'+discr),
        numerator = cms.vstring(meta_tagger),
        denominator = cms.vstring(),
        discrCut = cms.double(0.3),#Dummy,
        CTagPlots = cms.bool(False)
    )
    if "Bvs" in discr:
        UParTDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            bTagGenericAnalysisBlock
        )
    elif "Cvs" in discr:
        UParTDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            cTagGenericAnalysisBlock,
        )
        UParTDiscriminators[discr].CTagPlots = True
    elif "QvsG" in discr:
        UParTDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            cTagGenericAnalysisBlock
        )
    elif "Svs" in discr:
        UParTDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            cTagGenericAnalysisBlock
        )
    elif "TauVs" in discr:
        UParTDiscriminators[discr] = cms.PSet(
            commonTaggerConfig,
            cTagGenericAnalysisBlock
        )

