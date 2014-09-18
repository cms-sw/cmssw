import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.bTagAnalysisData_cfi import *
calobTagAnalysis = bTagAnalysis.clone()
bTagPlots = cms.Sequence(calobTagAnalysis)
calobTagAnalysis.finalizePlots = False
calobTagAnalysis.finalizeOnly = False


#Jet collection
JetCut=cms.string("neutralHadronEnergyFraction < 0.99 && neutralEmEnergyFraction < 0.99 && nConstituents > 1 && chargedHadronEnergyFraction > 0.0 && chargedMultiplicity > 0.0 && chargedEmEnergyFraction < 0.99")

from JetMETCorrections.Configuration.JetCorrectionServices_cff import ak4PFL2L3,ak4PFL2Relative,ak4PFL3Absolute
newAk4PFL2L3 = ak4PFL2L3.clone()

from JetMETCorrections.Configuration.DefaultJEC_cff import ak4PFJetsL2L3
ak4PFJetsJEC = ak4PFJetsL2L3.clone(
    correctors = ['newAk4PFL2L3']
    )

PFJetsFilter = cms.EDFilter("PFJetSelector",
                            src = cms.InputTag("ak4PFJetsJEC"),
                            cut = JetCut,
                            filter = cms.bool(False)
                            )

jetID = cms.InputTag("PFJetsFilter")

#JTA
from RecoJets.JetAssociationProducers.ak4JTA_cff import *
pfAk4JetTracksAssociatorAtVertex = ak4JetTracksAssociatorAtVertex.clone(jets = jetID)

#btag sequence
from RecoBTag.Configuration.RecoBTag_cff import *

pfImpactParameterTagInfos = impactParameterTagInfos.clone(jetTracks = cms.InputTag("pfAk4JetTracksAssociatorAtVertex"))
pfSecondaryVertexTagInfos = secondaryVertexTagInfos.clone(trackIPTagInfos = "pfImpactParameterTagInfos")

pfTrackCountingHighEffBJetTags = trackCountingHighEffBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos")))
pfTrackCountingHighPurBJetTags = trackCountingHighPurBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos")))

pfJetProbabilityBJetTags = jetProbabilityBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos")))
pfJetBProbabilityBJetTags = jetBProbabilityBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos")))

pfSimpleSecondaryVertexHighEffBJetTags = simpleSecondaryVertexHighEffBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexTagInfos")))
pfSimpleSecondaryVertexHighPurBJetTags = simpleSecondaryVertexHighPurBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSecondaryVertexTagInfos")))

pfGhostTrackVertexTagInfos = pfSecondaryVertexTagInfos.clone()
pfGhostTrackVertexTagInfos.vertexReco = ghostTrackVertexRecoBlock.vertexReco
pfGhostTrackVertexTagInfos.vertexCuts.multiplicityMin = 1
pfGhostTrackBJetTags = ghostTrackBJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                             cms.InputTag("pfGhostTrackVertexTagInfos"))
    )

pfCombinedSecondaryVertexBJetTags = combinedSecondaryVertexBJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                             cms.InputTag("pfSecondaryVertexTagInfos"))
    )
pfCombinedSecondaryVertexMVABJetTags = combinedSecondaryVertexMVABJetTags.clone(
    tagInfos = cms.VInputTag(cms.InputTag("pfImpactParameterTagInfos"),
                             cms.InputTag("pfSecondaryVertexTagInfos"))
    )

pfSoftPFMuonsTagInfos = softPFMuonsTagInfos.clone(jets = jetID)
pfSoftPFElectronsTagInfos = softPFElectronsTagInfos.clone(jets = jetID)
pfSoftPFMuonBJetTags = softPFMuonBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSoftPFMuonsTagInfos")))
pfSoftPFElectronBJetTags = softPFElectronBJetTags.clone(tagInfos = cms.VInputTag(cms.InputTag("pfSoftPFElectronsTagInfos")))

pfbtagging = cms.Sequence(
    pfImpactParameterTagInfos *
    ( pfTrackCountingHighEffBJetTags +
      pfTrackCountingHighPurBJetTags +
      pfJetProbabilityBJetTags +
      pfJetBProbabilityBJetTags +
      
      pfSecondaryVertexTagInfos *
      ( pfSimpleSecondaryVertexHighEffBJetTags +
        pfSimpleSecondaryVertexHighPurBJetTags +
        pfCombinedSecondaryVertexBJetTags +
        pfCombinedSecondaryVertexMVABJetTags
        ) +
      pfGhostTrackVertexTagInfos *
      pfGhostTrackBJetTags
      ) +
    
    #softPFLeptonsTagInfos*
    pfSoftPFMuonsTagInfos*
    pfSoftPFElectronsTagInfos*
    pfSoftPFElectronBJetTags*
    pfSoftPFMuonBJetTags
)


#preSeq
prebTagSequence = cms.Sequence(ak4PFJetsJEC*PFJetsFilter*pfAk4JetTracksAssociatorAtVertex*pfbtagging)
