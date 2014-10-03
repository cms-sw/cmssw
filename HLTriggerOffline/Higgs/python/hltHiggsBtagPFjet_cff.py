import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.bTagAnalysisData_cfi import *
#calobTagAnalysis = bTagAnalysis.clone()
#bTagPlots = cms.Sequence(calobTagAnalysis)
#calobTagAnalysis.finalizePlots = False
#calobTagAnalysis.finalizeOnly = False


#Jet collection
JetCut=cms.string("neutralHadronEnergyFraction < 0.99 && neutralEmEnergyFraction < 0.99 && nConstituents > 1 && chargedHadronEnergyFraction > 0.0 && chargedMultiplicity > 0.0 && chargedEmEnergyFraction < 0.99")

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFL2L3,ak4PFL2Relative,ak4PFL3Absolute
newAk4PFL2L3 = ak4PFL2L3.clone()

from JetMETCorrections.Configuration.DefaultJEC_cff import ak4PFJetsL2L3
ak4PFJetsJEC = ak4PFJetsL2L3.clone(
    correctors = ['newAk4PFL2L3']
    )

higgsDqmPFJetsFilter = cms.EDFilter("PFJetSelector",
                            src = cms.InputTag("ak4PFJetsJEC"),
                            cut = JetCut,
                            filter = cms.bool(False)
                            )

jetID = cms.InputTag("higgsDqmPFJetsFilter")

#JTA
from RecoJets.JetAssociationProducers.ak4JTA_cff import *
higgsDqmPfAk4JetTracksAssociatorAtVertex = ak4JetTracksAssociatorAtVertex.clone(jets = jetID)

#btag sequence
from RecoBTag.Configuration.RecoBTag_cff import *

hltHiggsDqmPfImpactParameterTagInfos = impactParameterTagInfos.clone(jetTracks = cms.InputTag("higgsDqmPfAk4JetTracksAssociatorAtVertex"))
hltHiggsDqmPfSecondaryVertexTagInfos = secondaryVertexTagInfos.clone(trackIPTagInfos = "hltHiggsDqmPfImpactParameterTagInfos")

hltHiggsDqmPfTrackCountingHighEffBJetTags = trackCountingHighEffBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfImpactParameterTagInfos")))
hltHiggsDqmPfTrackCountingHighPurBJetTags = trackCountingHighPurBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfImpactParameterTagInfos")))

hltHiggsDqmPfJetProbabilityBJetTags = jetProbabilityBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfImpactParameterTagInfos")))
hltHiggsDqmPfJetBProbabilityBJetTags = jetBProbabilityBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfImpactParameterTagInfos")))

hltHiggsDqmPfSimpleSecondaryVertexHighEffBJetTags = simpleSecondaryVertexHighEffBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfSecondaryVertexTagInfos")))
hltHiggsDqmPfSimpleSecondaryVertexHighPurBJetTags = simpleSecondaryVertexHighPurBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfSecondaryVertexTagInfos")))

hltHiggsDqmPfGhostTrackVertexTagInfos = hltHiggsDqmPfSecondaryVertexTagInfos.clone()
hltHiggsDqmPfGhostTrackVertexTagInfos.vertexReco = ghostTrackVertexRecoBlock.vertexReco
hltHiggsDqmPfGhostTrackVertexTagInfos.vertexCuts.multiplicityMin = 1
hltHiggsDqmPfGhostTrackBJetTags = ghostTrackBJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfImpactParameterTagInfos"),
                             cms.InputTag("hltHiggsDqmPfGhostTrackVertexTagInfos"))
    )

hltHiggsDqmPfCombinedSecondaryVertexBJetTags = combinedSecondaryVertexBJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfImpactParameterTagInfos"),
                             cms.InputTag("hltHiggsDqmPfSecondaryVertexTagInfos"))
    )
hltHiggsDqmPfCombinedSecondaryVertexMVABJetTags = combinedSecondaryVertexMVABJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfImpactParameterTagInfos"),
                             cms.InputTag("hltHiggsDqmPfSecondaryVertexTagInfos"))
    )

hltHiggsDqmPfSoftPFMuonsTagInfos = softPFMuonsTagInfos.clone(jets = jetID)
hltHiggsDqmPfSoftPFElectronsTagInfos = softPFElectronsTagInfos.clone(jets = jetID)
hltHiggsDqmPfSoftPFMuonBJetTags = softPFMuonBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfSoftPFMuonsTagInfos")))
hltHiggsDqmPfSoftPFElectronBJetTags = softPFElectronBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("hltHiggsDqmPfSoftPFElectronsTagInfos")))

pfbtagging = cms.Sequence(
    hltHiggsDqmPfImpactParameterTagInfos *
    ( hltHiggsDqmPfTrackCountingHighEffBJetTags +
      hltHiggsDqmPfTrackCountingHighPurBJetTags +
      hltHiggsDqmPfJetProbabilityBJetTags +
      hltHiggsDqmPfJetBProbabilityBJetTags +
      
      hltHiggsDqmPfSecondaryVertexTagInfos *
      ( hltHiggsDqmPfSimpleSecondaryVertexHighEffBJetTags +
        hltHiggsDqmPfSimpleSecondaryVertexHighPurBJetTags +
        hltHiggsDqmPfCombinedSecondaryVertexBJetTags +
        hltHiggsDqmPfCombinedSecondaryVertexMVABJetTags
        ) +
      hltHiggsDqmPfGhostTrackVertexTagInfos *
      hltHiggsDqmPfGhostTrackBJetTags
      ) +
    
    #softPFLeptonsTagInfos*
    hltHiggsDqmPfSoftPFMuonsTagInfos*
    hltHiggsDqmPfSoftPFElectronsTagInfos*
    hltHiggsDqmPfSoftPFElectronBJetTags*
    hltHiggsDqmPfSoftPFMuonBJetTags
)

#preSeq
bTagForHiggsDqm = cms.Sequence(ak4PFJetsJEC*higgsDqmPFJetsFilter*higgsDqmPfAk4JetTracksAssociatorAtVertex*pfbtagging)
