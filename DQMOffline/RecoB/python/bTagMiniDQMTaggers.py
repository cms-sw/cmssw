import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.tagGenericAnalysis_cff import bTagGenericAnalysisBlock
from DQMOffline.RecoB.tagGenericAnalysis_cff import cTagGenericAnalysisBlock
from DQMOffline.RecoB.tagGenericAnalysis_cff import tauTagGenericAnalysisBlock
from DQMOffline.RecoB.tagGenericAnalysis_cff import sTagGenericAnalysisBlock
from DQMOffline.RecoB.tagGenericAnalysis_cff import qgTagGenericAnalysisBlock

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

