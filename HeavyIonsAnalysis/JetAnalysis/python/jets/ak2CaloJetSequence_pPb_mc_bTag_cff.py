

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak2CaloJets"),
    matched = cms.InputTag("ak2HiGenJets")
    )

ak2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak2CaloJets")
                                                        )

ak2Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak2CaloJets"),
    payload = "AK2Calo_HI"
    )

ak2CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak2CaloJets'))

ak2Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJets'))

ak2CalobTagger = bTaggers("ak2Calo")

#create objects locally since they dont load properly otherwise
ak2Calomatch = ak2CalobTagger.match
ak2Caloparton = ak2CalobTagger.parton
ak2CaloPatJetFlavourAssociation = ak2CalobTagger.PatJetFlavourAssociation
ak2CaloJetTracksAssociatorAtVertex = ak2CalobTagger.JetTracksAssociatorAtVertex
ak2CaloSimpleSecondaryVertexHighEffBJetTags = ak2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak2CaloSimpleSecondaryVertexHighPurBJetTags = ak2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak2CaloCombinedSecondaryVertexBJetTags = ak2CalobTagger.CombinedSecondaryVertexBJetTags
ak2CaloCombinedSecondaryVertexMVABJetTags = ak2CalobTagger.CombinedSecondaryVertexMVABJetTags
ak2CaloJetBProbabilityBJetTags = ak2CalobTagger.JetBProbabilityBJetTags
ak2CaloSoftMuonByPtBJetTags = ak2CalobTagger.SoftMuonByPtBJetTags
ak2CaloSoftMuonByIP3dBJetTags = ak2CalobTagger.SoftMuonByIP3dBJetTags
ak2CaloTrackCountingHighEffBJetTags = ak2CalobTagger.TrackCountingHighEffBJetTags
ak2CaloTrackCountingHighPurBJetTags = ak2CalobTagger.TrackCountingHighPurBJetTags
ak2CaloPatJetPartonAssociation = ak2CalobTagger.PatJetPartonAssociation

ak2CaloImpactParameterTagInfos = ak2CalobTagger.ImpactParameterTagInfos
ak2CaloJetProbabilityBJetTags = ak2CalobTagger.JetProbabilityBJetTags
ak2CaloPositiveOnlyJetProbabilityJetTags = ak2CalobTagger.PositiveOnlyJetProbabilityJetTags
ak2CaloNegativeOnlyJetProbabilityJetTags = ak2CalobTagger.NegativeOnlyJetProbabilityJetTags
ak2CaloNegativeTrackCountingHighEffJetTags = ak2CalobTagger.NegativeTrackCountingHighEffJetTags
ak2CaloNegativeTrackCountingHighPur = ak2CalobTagger.NegativeTrackCountingHighPur
ak2CaloNegativeOnlyJetBProbabilityJetTags = ak2CalobTagger.NegativeOnlyJetBProbabilityJetTags
ak2CaloPositiveOnlyJetBProbabilityJetTags = ak2CalobTagger.PositiveOnlyJetBProbabilityJetTags

ak2CaloSecondaryVertexTagInfos = ak2CalobTagger.SecondaryVertexTagInfos
ak2CaloSimpleSecondaryVertexHighEffBJetTags = ak2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak2CaloSimpleSecondaryVertexHighPurBJetTags = ak2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak2CaloCombinedSecondaryVertexBJetTags = ak2CalobTagger.CombinedSecondaryVertexBJetTags
ak2CaloCombinedSecondaryVertexMVABJetTags = ak2CalobTagger.CombinedSecondaryVertexMVABJetTags

ak2CaloSecondaryVertexNegativeTagInfos = ak2CalobTagger.SecondaryVertexNegativeTagInfos
ak2CaloSimpleSecondaryVertexNegativeHighEffBJetTags = ak2CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak2CaloSimpleSecondaryVertexNegativeHighPurBJetTags = ak2CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak2CaloCombinedSecondaryVertexNegativeBJetTags = ak2CalobTagger.CombinedSecondaryVertexNegativeBJetTags
ak2CaloCombinedSecondaryVertexPositiveBJetTags = ak2CalobTagger.CombinedSecondaryVertexPositiveBJetTags

ak2CaloSoftMuonTagInfos = ak2CalobTagger.SoftMuonTagInfos
ak2CaloSoftMuonBJetTags = ak2CalobTagger.SoftMuonBJetTags
ak2CaloSoftMuonByIP3dBJetTags = ak2CalobTagger.SoftMuonByIP3dBJetTags
ak2CaloSoftMuonByPtBJetTags = ak2CalobTagger.SoftMuonByPtBJetTags
ak2CaloNegativeSoftMuonByPtBJetTags = ak2CalobTagger.NegativeSoftMuonByPtBJetTags
ak2CaloPositiveSoftMuonByPtBJetTags = ak2CalobTagger.PositiveSoftMuonByPtBJetTags

ak2CaloPatJetFlavourId = cms.Sequence(ak2CaloPatJetPartonAssociation*ak2CaloPatJetFlavourAssociation)

ak2CaloJetBtaggingIP       = cms.Sequence(ak2CaloImpactParameterTagInfos *
            (ak2CaloTrackCountingHighEffBJetTags +
             ak2CaloTrackCountingHighPurBJetTags +
             ak2CaloJetProbabilityBJetTags +
             ak2CaloJetBProbabilityBJetTags +
             ak2CaloPositiveOnlyJetProbabilityJetTags +
             ak2CaloNegativeOnlyJetProbabilityJetTags +
             ak2CaloNegativeTrackCountingHighEffJetTags +
             ak2CaloNegativeTrackCountingHighPur +
             ak2CaloNegativeOnlyJetBProbabilityJetTags +
             ak2CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

ak2CaloJetBtaggingSV = cms.Sequence(ak2CaloImpactParameterTagInfos
            *
            ak2CaloSecondaryVertexTagInfos
            * (ak2CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak2CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak2CaloCombinedSecondaryVertexBJetTags
                +
                ak2CaloCombinedSecondaryVertexMVABJetTags
              )
            )

ak2CaloJetBtaggingNegSV = cms.Sequence(ak2CaloImpactParameterTagInfos
            *
            ak2CaloSecondaryVertexNegativeTagInfos
            * (ak2CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak2CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak2CaloCombinedSecondaryVertexNegativeBJetTags
                +
                ak2CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak2CaloJetBtaggingMu = cms.Sequence(ak2CaloSoftMuonTagInfos * (ak2CaloSoftMuonBJetTags
                +
                ak2CaloSoftMuonByIP3dBJetTags
                +
                ak2CaloSoftMuonByPtBJetTags
                +
                ak2CaloNegativeSoftMuonByPtBJetTags
                +
                ak2CaloPositiveSoftMuonByPtBJetTags
              )
            )

ak2CaloJetBtagging = cms.Sequence(ak2CaloJetBtaggingIP
            *ak2CaloJetBtaggingSV
            *ak2CaloJetBtaggingNegSV
            *ak2CaloJetBtaggingMu
            )

ak2CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak2CaloJets"),
        genJetMatch          = cms.InputTag("ak2Calomatch"),
        genPartonMatch       = cms.InputTag("ak2Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak2Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak2CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak2CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak2CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak2CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak2CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak2CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak2CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak2CaloJetProbabilityBJetTags"),
            cms.InputTag("ak2CaloSoftMuonByPtBJetTags"),
            cms.InputTag("ak2CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak2CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak2CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak2CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = True,
        getJetMCFlavour = True,
        addGenPartonMatch = True,
        addGenJetMatch = True,
        embedGenJetMatch = True,
        embedGenPartonMatch = True,
        embedCaloTowers = False,
        embedPFCandidates = True
        )

ak2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak2CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("ak2Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak2CaloJetSequence_mc = cms.Sequence(
                                                  ak2Caloclean
                                                  *
                                                  ak2Calomatch
                                                  *
                                                  ak2Caloparton
                                                  *
                                                  ak2Calocorr
                                                  *
                                                  ak2CaloJetID
                                                  *
                                                  ak2CaloPatJetFlavourId
                                                  *
                                                  ak2CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak2CaloJetBtagging
                                                  *
                                                  ak2CalopatJetsWithBtagging
                                                  *
                                                  ak2CaloJetAnalyzer
                                                  )

ak2CaloJetSequence_data = cms.Sequence(ak2Calocorr
                                                    *
                                                    ak2CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak2CaloJetBtagging
                                                    *
                                                    ak2CalopatJetsWithBtagging
                                                    *
                                                    ak2CaloJetAnalyzer
                                                    )

ak2CaloJetSequence_jec = ak2CaloJetSequence_mc
ak2CaloJetSequence_mix = ak2CaloJetSequence_mc

ak2CaloJetSequence = cms.Sequence(ak2CaloJetSequence_mc)
