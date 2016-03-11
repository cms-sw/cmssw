import FWCore.ParameterSet.Config as cms

#from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import *
from PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff import *


#load all the b-tagging algorithms
from RecoBTag.SecondaryVertex.secondaryVertexNegativeTagInfos_cfi import *
from RecoBTag.SecondaryVertex.negativeSimpleSecondaryVertexHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeSimpleSecondaryVertexHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeCombinedSecondaryVertexComputer_cfi import *
from RecoBTag.SecondaryVertex.negativeCombinedSecondaryVertexBJetTags_cfi import *
from RecoBTag.SecondaryVertex.positiveCombinedSecondaryVertexComputer_cfi import *
from RecoBTag.SecondaryVertex.positiveCombinedSecondaryVertexBJetTags_cfi import *
from RecoBTag.SecondaryVertex.negativeCombinedSecondaryVertexV2Computer_cfi import *
from RecoBTag.SecondaryVertex.negativeCombinedSecondaryVertexV2BJetTags_cfi import *
from RecoBTag.SecondaryVertex.positiveCombinedSecondaryVertexV2Computer_cfi import *
from RecoBTag.SecondaryVertex.positiveCombinedSecondaryVertexV2BJetTags_cfi import *

from RecoJets.JetAssociationProducers.ak5JTA_cff import *
from RecoBTag.Configuration.RecoBTag_cff import *

class bTaggers:
    def __init__(self,jetname,rParam):
        self.JetTracksAssociatorAtVertex = ak5JetTracksAssociatorAtVertex.clone()
        self.JetTracksAssociatorAtVertex.jets = cms.InputTag(jetname+"Jets")
        self.JetTracksAssociatorAtVertex.tracks = cms.InputTag("hiGeneralTracks")
        self.ImpactParameterTagInfos = impactParameterTagInfos.clone()
        self.ImpactParameterTagInfos.jetTracks = cms.InputTag(jetname+"JetTracksAssociatorAtVertex")
        self.ImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
        self.TrackCountingHighEffBJetTags          = trackCountingHighEffBJetTags.clone()
        self.TrackCountingHighEffBJetTags.tagInfos = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))
        self.TrackCountingHighPurBJetTags          = trackCountingHighPurBJetTags.clone()
        self.TrackCountingHighPurBJetTags.tagInfos = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))
        self.JetProbabilityBJetTags                = jetProbabilityBJetTags.clone()
        self.JetProbabilityBJetTags.tagInfos       = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))
        self.JetBProbabilityBJetTags               = jetBProbabilityBJetTags.clone()
        self.JetBProbabilityBJetTags.tagInfos      = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))

        self.SecondaryVertexTagInfos                     = secondaryVertexTagInfos.clone()
        self.SecondaryVertexTagInfos.trackIPTagInfos     = cms.InputTag(jetname+"ImpactParameterTagInfos")
        #self.SimpleSecondaryVertexBJetTags               = simpleSecondaryVertexBJetTags.clone()
        #self.SimpleSecondaryVertexBJetTags.tagInfos      = cms.VInputTag(cms.InputTag(jetname+"SecondaryVertexTagInfos"))
        self.CombinedSecondaryVertexBJetTags             = combinedSecondaryVertexBJetTags.clone()
        self.CombinedSecondaryVertexBJetTags.tagInfos    = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                cms.InputTag(jetname+"SecondaryVertexTagInfos"))
        self.CombinedSecondaryVertexV2BJetTags          = combinedSecondaryVertexV2BJetTags.clone()
        self.CombinedSecondaryVertexV2BJetTags.tagInfos = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                cms.InputTag(jetname+"SecondaryVertexTagInfos"))


        # secondary vertex b-tag
        self.SecondaryVertexTagInfos                     = secondaryVertexTagInfos.clone()
        self.SecondaryVertexTagInfos.trackIPTagInfos     = cms.InputTag(jetname+"ImpactParameterTagInfos")
        self.SimpleSecondaryVertexHighEffBJetTags               = simpleSecondaryVertexHighEffBJetTags.clone()
        self.SimpleSecondaryVertexHighEffBJetTags.tagInfos      = cms.VInputTag(cms.InputTag(jetname+"SecondaryVertexTagInfos"))
        self.SimpleSecondaryVertexHighPurBJetTags               = simpleSecondaryVertexHighPurBJetTags.clone()
        self.SimpleSecondaryVertexHighPurBJetTags.tagInfos      = cms.VInputTag(cms.InputTag(jetname+"SecondaryVertexTagInfos"))
        self.CombinedSecondaryVertexBJetTags             = combinedSecondaryVertexBJetTags.clone()
        self.CombinedSecondaryVertexBJetTags.tagInfos    = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                cms.InputTag(jetname+"SecondaryVertexTagInfos"))
        self.CombinedSecondaryVertexV2BJetTags          = combinedSecondaryVertexV2BJetTags.clone()
        self.CombinedSecondaryVertexV2BJetTags.tagInfos = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                cms.InputTag(jetname+"SecondaryVertexTagInfos"))

        self.SecondaryVertexNegativeTagInfos                     = secondaryVertexNegativeTagInfos.clone()
        self.SecondaryVertexNegativeTagInfos.trackIPTagInfos     = cms.InputTag(jetname+"ImpactParameterTagInfos")

        self.NegativeSimpleSecondaryVertexHighEffBJetTags               = negativeSimpleSecondaryVertexHighEffBJetTags.clone()
        self.NegativeSimpleSecondaryVertexHighEffBJetTags.tagInfos      = cms.VInputTag(cms.InputTag(jetname+"SecondaryVertexNegativeTagInfos"))
        self.NegativeSimpleSecondaryVertexHighPurBJetTags               = negativeSimpleSecondaryVertexHighPurBJetTags.clone()
        self.NegativeSimpleSecondaryVertexHighPurBJetTags.tagInfos      = cms.VInputTag(cms.InputTag(jetname+"SecondaryVertexNegativeTagInfos"))

        self.NegativeCombinedSecondaryVertexBJetTags                    = negativeCombinedSecondaryVertexBJetTags.clone()
        self.NegativeCombinedSecondaryVertexBJetTags.tagInfos    = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                                                                                 cms.InputTag(jetname+"SecondaryVertexNegativeTagInfos"))
        self.PositiveCombinedSecondaryVertexBJetTags                    = positiveCombinedSecondaryVertexBJetTags.clone()
        self.PositiveCombinedSecondaryVertexBJetTags.tagInfos    = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                                                                                 cms.InputTag(jetname+"SecondaryVertexTagInfos"))

        self.NegativeCombinedSecondaryVertexV2BJetTags                    = negativeCombinedSecondaryVertexV2BJetTags.clone()
        self.NegativeCombinedSecondaryVertexV2BJetTags.tagInfos    = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                                                                                 cms.InputTag(jetname+"SecondaryVertexNegativeTagInfos"))
        self.PositiveCombinedSecondaryVertexV2BJetTags                    = positiveCombinedSecondaryVertexV2BJetTags.clone()
        self.PositiveCombinedSecondaryVertexV2BJetTags.tagInfos    = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                                                                                 cms.InputTag(jetname+"SecondaryVertexTagInfos"))

        self.SoftPFMuonsTagInfos                = softPFMuonsTagInfos.clone()
        self.SoftPFMuonsTagInfos.jets           = cms.InputTag(jetname+"Jets")
        #self.SoftPFMuonsTagInfos.primaryVertex  = cms.InputTag("offlinePrimaryVertices")
        self.SoftPFMuonBJetTags                = softPFMuonBJetTags.clone()
        self.SoftPFMuonBJetTags.tagInfos       = cms.VInputTag(cms.InputTag(jetname+"SoftPFMuonsTagInfos"))
        self.SoftPFMuonByIP3dBJetTags          = softPFMuonByIP3dBJetTags.clone()
        self.SoftPFMuonByIP3dBJetTags.tagInfos = cms.VInputTag(cms.InputTag(jetname+"SoftPFMuonsTagInfos"))
        self.SoftPFMuonByPtBJetTags            = softPFMuonByPtBJetTags.clone()
        self.SoftPFMuonByPtBJetTags.tagInfos   = cms.VInputTag(cms.InputTag(jetname+"SoftPFMuonsTagInfos"))

        self.PositiveSoftPFMuonByPtBJetTags                = positiveSoftPFMuonByPtBJetTags.clone()
        self.PositiveSoftPFMuonByPtBJetTags.tagInfos       = cms.VInputTag(cms.InputTag(jetname+"SoftPFMuonsTagInfos"))

        # soft muon negative taggers
        self.NegativeSoftPFMuonByPtBJetTags                = negativeSoftPFMuonByPtBJetTags.clone()
        self.NegativeSoftPFMuonByPtBJetTags.tagInfos       = cms.VInputTag(cms.InputTag(jetname+"SoftPFMuonsTagInfos"))

        self.JetTracksAssociator = cms.Sequence(self.JetTracksAssociatorAtVertex)
        self.JetBtaggingIP       = cms.Sequence(self.ImpactParameterTagInfos * (
                self.TrackCountingHighEffBJetTags +
                self.TrackCountingHighPurBJetTags +
                self.JetProbabilityBJetTags +
                self.JetBProbabilityBJetTags 
                )
                                                )

        self.JetBtaggingSV = cms.Sequence(self.ImpactParameterTagInfos *
                self.SecondaryVertexTagInfos * (self.SimpleSecondaryVertexHighEffBJetTags +
                    self.SimpleSecondaryVertexHighPurBJetTags +
                    self.CombinedSecondaryVertexBJetTags +
                    self.CombinedSecondaryVertexV2BJetTags
                    )
                )

        self.JetBtaggingNegSV = cms.Sequence(self.ImpactParameterTagInfos *
                self.SecondaryVertexNegativeTagInfos * (self.NegativeSimpleSecondaryVertexHighEffBJetTags +
                    self.NegativeSimpleSecondaryVertexHighPurBJetTags +
                    self.NegativeCombinedSecondaryVertexBJetTags +
                    self.PositiveCombinedSecondaryVertexBJetTags +
                    self.NegativeCombinedSecondaryVertexV2BJetTags +
                    self.PositiveCombinedSecondaryVertexV2BJetTags
                    )
                )


        self.JetBtaggingMu = cms.Sequence(self.SoftPFMuonsTagInfos 
                                          * (self.SoftPFMuonBJetTags +
                                             self.SoftPFMuonByIP3dBJetTags +
                                             self.SoftPFMuonByPtBJetTags +
                                             self.NegativeSoftPFMuonByPtBJetTags +
                                             self.PositiveSoftPFMuonByPtBJetTags
            )
                                          )

        self.JetBtagging = cms.Sequence(self.JetBtaggingIP
                *self.JetBtaggingSV
                *self.JetBtaggingNegSV
                *self.JetBtaggingMu
                )

        self.PatJetPartonAssociationLegacy       = patJetPartonAssociationLegacy.clone(
            jets = cms.InputTag(jetname+"Jets"),
            partons = cms.InputTag("myPartons")
            )

        self.PatJetFlavourAssociationLegacy      = patJetFlavourAssociationLegacy.clone(
            srcByReference = cms.InputTag(jetname+"PatJetPartonAssociationLegacy")
            )

        self.patJetFlavourIdLegacy = cms.Sequence( self.PatJetPartonAssociationLegacy * self.PatJetFlavourAssociationLegacy)

        self.PatJetPartons = patJetPartons.clone()
        self.PatJetFlavourAssociation = patJetFlavourAssociation.clone(
            jets = cms.InputTag(jetname+"Jets"),
            rParam = rParam,
            bHadrons = cms.InputTag(jetname+"PatJetPartons","bHadrons"),
            cHadrons = cms.InputTag(jetname+"PatJetPartons","cHadrons"),
            partons = cms.InputTag(jetname+"PatJetPartons","partons")
            )

        self.PatJetFlavourId               = cms.Sequence(self.PatJetPartons*self.PatJetFlavourAssociation)
        #self.match   = patJetGenJetMatch.clone(
        #    src      = cms.InputTag(jetname+"Jets"),
        #    matched  = cms.InputTag(jetname+"clean"),
        #    maxDeltaR = rParam 
        #    )
        self.parton  = patJetPartonMatch.clone(src      = cms.InputTag(jetname+"Jets"),
                                                matched = cms.InputTag("genParticles")
                                                )

