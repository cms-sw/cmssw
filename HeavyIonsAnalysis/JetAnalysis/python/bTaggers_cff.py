import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *

#load all the b-tagging algorithms
from RecoBTag.ImpactParameter.positiveOnlyJetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.positiveOnlyJetProbabilityJetTags_cfi import *
from RecoBTag.ImpactParameter.negativeOnlyJetProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.negativeOnlyJetProbabilityJetTags_cfi import *
from RecoBTag.ImpactParameter.negativeTrackCountingHighEffJetTags_cfi import *
from RecoBTag.ImpactParameter.negativeTrackCountingHighPur_cfi import *
from RecoBTag.ImpactParameter.positiveOnlyJetBProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.positiveOnlyJetBProbabilityJetTags_cfi import *
from RecoBTag.ImpactParameter.negativeOnlyJetBProbabilityComputer_cfi import *
from RecoBTag.ImpactParameter.negativeOnlyJetBProbabilityJetTags_cfi import *
from RecoBTag.SecondaryVertex.secondaryVertexNegativeTagInfos_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertexNegativeHighEffBJetTags_cfi import *
from RecoBTag.SecondaryVertex.simpleSecondaryVertexNegativeHighPurBJetTags_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexNegativeES_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexNegativeBJetTags_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexPositiveES_cfi import *
from RecoBTag.SecondaryVertex.combinedSecondaryVertexPositiveBJetTags_cfi import *

from RecoJets.JetAssociationProducers.ak5JTA_cff import *
from RecoBTag.Configuration.RecoBTag_cff import *

class bTaggers:
    def __init__(self,jetname):
        self.JetTracksAssociatorAtVertex = ak5JetTracksAssociatorAtVertex.clone()
        self.JetTracksAssociatorAtVertex.jets = cms.InputTag(jetname+"Jets")
        self.JetTracksAssociatorAtVertex.tracks = cms.InputTag("generalTracks")
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
        self.SimpleSecondaryVertexBJetTags               = simpleSecondaryVertexBJetTags.clone()
        self.SimpleSecondaryVertexBJetTags.tagInfos      = cms.VInputTag(cms.InputTag(jetname+"SecondaryVertexTagInfos"))
        self.CombinedSecondaryVertexBJetTags             = combinedSecondaryVertexBJetTags.clone()
        self.CombinedSecondaryVertexBJetTags.tagInfos    = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                cms.InputTag(jetname+"SecondaryVertexTagInfos"))
        self.CombinedSecondaryVertexMVABJetTags          = combinedSecondaryVertexMVABJetTags.clone()
        self.CombinedSecondaryVertexMVABJetTags.tagInfos = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                cms.InputTag(jetname+"SecondaryVertexTagInfos"))

        self.PositiveOnlyJetProbabilityJetTags     =       positiveOnlyJetProbabilityJetTags.clone()
        self.NegativeOnlyJetProbabilityJetTags     =       negativeOnlyJetProbabilityJetTags.clone()
        self.NegativeTrackCountingHighEffJetTags   =       negativeTrackCountingHighEffJetTags.clone()
        self.NegativeTrackCountingHighPur          =       negativeTrackCountingHighPur.clone()
        self.NegativeOnlyJetBProbabilityJetTags    =       negativeOnlyJetBProbabilityJetTags.clone()
        self.PositiveOnlyJetBProbabilityJetTags    =       positiveOnlyJetBProbabilityJetTags.clone()

        self.PositiveOnlyJetProbabilityJetTags.tagInfos   = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))
        self.NegativeOnlyJetProbabilityJetTags.tagInfos   = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))
        self.NegativeTrackCountingHighEffJetTags.tagInfos = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))
        self.NegativeTrackCountingHighPur.tagInfos        = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))
        self.NegativeOnlyJetBProbabilityJetTags.tagInfos  = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))
        self.PositiveOnlyJetBProbabilityJetTags.tagInfos  = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"))


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
        self.CombinedSecondaryVertexMVABJetTags          = combinedSecondaryVertexMVABJetTags.clone()
        self.CombinedSecondaryVertexMVABJetTags.tagInfos = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                cms.InputTag(jetname+"SecondaryVertexTagInfos"))

        self.SecondaryVertexNegativeTagInfos                     = secondaryVertexNegativeTagInfos.clone()
        self.SecondaryVertexNegativeTagInfos.trackIPTagInfos     = cms.InputTag(jetname+"ImpactParameterTagInfos")
        self.SimpleSecondaryVertexNegativeHighEffBJetTags               = simpleSecondaryVertexNegativeHighEffBJetTags.clone()
        self.SimpleSecondaryVertexNegativeHighEffBJetTags.tagInfos      = cms.VInputTag(cms.InputTag(jetname+"SecondaryVertexNegativeTagInfos"))
        self.SimpleSecondaryVertexNegativeHighPurBJetTags               = simpleSecondaryVertexNegativeHighPurBJetTags.clone()
        self.SimpleSecondaryVertexNegativeHighPurBJetTags.tagInfos      = cms.VInputTag(cms.InputTag(jetname+"SecondaryVertexNegativeTagInfos"))
        self.CombinedSecondaryVertexNegativeBJetTags                    = combinedSecondaryVertexNegativeBJetTags.clone()
        self.CombinedSecondaryVertexNegativeBJetTags.tagInfos    = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                cms.InputTag(jetname+"SecondaryVertexNegativeTagInfos"))
        self.CombinedSecondaryVertexPositiveBJetTags                    = combinedSecondaryVertexPositiveBJetTags.clone()
        self.CombinedSecondaryVertexPositiveBJetTags.tagInfos    = cms.VInputTag(cms.InputTag(jetname+"ImpactParameterTagInfos"),
                cms.InputTag(jetname+"SecondaryVertexTagInfos"))

        self.SoftMuonTagInfos                = softMuonTagInfos.clone()
        self.SoftMuonTagInfos.jets           = cms.InputTag(jetname+"Jets")
        self.SoftMuonTagInfos.primaryVertex  = cms.InputTag("offlinePrimaryVertices")
        self.SoftMuonBJetTags                = softPFMuonBJetTags.clone()
        self.SoftMuonBJetTags.tagInfos       = cms.VInputTag(cms.InputTag(jetname+"SoftMuonTagInfos"))
        self.SoftMuonByIP3dBJetTags          = softPFMuonByIP3dBJetTags.clone()
        self.SoftMuonByIP3dBJetTags.tagInfos = cms.VInputTag(cms.InputTag(jetname+"SoftMuonTagInfos"))
        self.SoftMuonByPtBJetTags            = softPFMuonByPtBJetTags.clone()
        self.SoftMuonByPtBJetTags.tagInfos   = cms.VInputTag(cms.InputTag(jetname+"SoftMuonTagInfos"))

        self.PositiveSoftMuonByPtBJetTags                = positiveSoftPFMuonByPtBJetTags.clone()
        self.PositiveSoftMuonByPtBJetTags.tagInfos       = cms.VInputTag(cms.InputTag(jetname+"SoftMuonTagInfos"))

        # soft muon negative taggers
        self.NegativeSoftMuonByPtBJetTags                = negativeSoftPFMuonByPtBJetTags.clone()
        self.NegativeSoftMuonByPtBJetTags.tagInfos       = cms.VInputTag(cms.InputTag(jetname+"SoftMuonTagInfos"))

        self.JetTracksAssociator = cms.Sequence(self.JetTracksAssociatorAtVertex)
        self.JetBtaggingIP       = cms.Sequence(self.ImpactParameterTagInfos * (self.TrackCountingHighEffBJetTags +
            self.TrackCountingHighPurBJetTags +
            self.JetProbabilityBJetTags +
            self.JetBProbabilityBJetTags +
            self.PositiveOnlyJetProbabilityJetTags +
            self.NegativeOnlyJetProbabilityJetTags +
            self.NegativeTrackCountingHighEffJetTags +
            self.NegativeTrackCountingHighPur +
            self.NegativeOnlyJetBProbabilityJetTags +
            self.PositiveOnlyJetBProbabilityJetTags
            )
            )

        self.JetBtaggingSV = cms.Sequence(self.ImpactParameterTagInfos *
                self.SecondaryVertexTagInfos * (self.SimpleSecondaryVertexHighEffBJetTags +
                    self.SimpleSecondaryVertexHighPurBJetTags +
                    self.CombinedSecondaryVertexBJetTags +
                    self.CombinedSecondaryVertexMVABJetTags
                    )
                )

        self.JetBtaggingNegSV = cms.Sequence(self.ImpactParameterTagInfos *
                self.SecondaryVertexNegativeTagInfos * (self.SimpleSecondaryVertexNegativeHighEffBJetTags +
                    self.SimpleSecondaryVertexNegativeHighPurBJetTags +
                    self.CombinedSecondaryVertexNegativeBJetTags +
                    self.CombinedSecondaryVertexPositiveBJetTags
                    )
                )


        self.JetBtaggingMu = cms.Sequence(self.SoftMuonTagInfos * (self.SoftMuonBJetTags +
            self.SoftMuonByIP3dBJetTags +
            self.SoftMuonByPtBJetTags +
            self.NegativeSoftMuonByPtBJetTags +
            self.PositiveSoftMuonByPtBJetTags
            )
            )

        self.JetBtagging = cms.Sequence(self.JetBtaggingIP
                *self.JetBtaggingSV
                *self.JetBtaggingNegSV
                *self.JetBtaggingMu
                )

        self.PatJetPartonAssociation       = patJetPartonAssociationLegacy.clone(jets = cms.InputTag(jetname+"Jets"),
                partons = cms.InputTag("genPartons"),
                coneSizeToAssociate = cms.double(0.4))

        self.PatJetFlavourAssociation      = patJetFlavourAssociationLegacy.clone(srcByReference = cms.InputTag(jetname+"PatJetPartonAssociation"))

        self.PatJetFlavourId               = cms.Sequence(self.PatJetPartonAssociation*self.PatJetFlavourAssociation)
        self.match   = patJetGenJetMatch.clone(src      = cms.InputTag(jetname+"Jets"),
                matched  = cms.InputTag(jetname+"clean"))
        self.parton  = patJetPartonMatch.clone(src      = cms.InputTag(jetname+"Jets"),
                                                matched = cms.InputTag("hiPartons")
                                                )

