

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak7Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak7CaloJets"),
    matched = cms.InputTag("ak7HiGenJets")
    )

ak7Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak7CaloJets")
                                                        )

ak7Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak7CaloJets"),
    payload = "AK7Calo_HI"
    )

ak7CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak7CaloJets'))

ak7Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

ak7CalobTagger = bTaggers("ak7Calo")

#create objects locally since they dont load properly otherwise
ak7Calomatch = ak7CalobTagger.match
ak7Caloparton = ak7CalobTagger.parton
ak7CaloPatJetFlavourAssociation = ak7CalobTagger.PatJetFlavourAssociation
ak7CaloJetTracksAssociatorAtVertex = ak7CalobTagger.JetTracksAssociatorAtVertex
ak7CaloSimpleSecondaryVertexHighEffBJetTags = ak7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak7CaloSimpleSecondaryVertexHighPurBJetTags = ak7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak7CaloCombinedSecondaryVertexBJetTags = ak7CalobTagger.CombinedSecondaryVertexBJetTags
ak7CaloCombinedSecondaryVertexMVABJetTags = ak7CalobTagger.CombinedSecondaryVertexMVABJetTags
ak7CaloJetBProbabilityBJetTags = ak7CalobTagger.JetBProbabilityBJetTags
ak7CaloSoftMuonByPtBJetTags = ak7CalobTagger.SoftMuonByPtBJetTags
ak7CaloSoftMuonByIP3dBJetTags = ak7CalobTagger.SoftMuonByIP3dBJetTags
ak7CaloTrackCountingHighEffBJetTags = ak7CalobTagger.TrackCountingHighEffBJetTags
ak7CaloTrackCountingHighPurBJetTags = ak7CalobTagger.TrackCountingHighPurBJetTags
ak7CaloPatJetPartonAssociation = ak7CalobTagger.PatJetPartonAssociation

ak7CaloImpactParameterTagInfos = ak7CalobTagger.ImpactParameterTagInfos
ak7CaloJetProbabilityBJetTags = ak7CalobTagger.JetProbabilityBJetTags
ak7CaloPositiveOnlyJetProbabilityJetTags = ak7CalobTagger.PositiveOnlyJetProbabilityJetTags
ak7CaloNegativeOnlyJetProbabilityJetTags = ak7CalobTagger.NegativeOnlyJetProbabilityJetTags
ak7CaloNegativeTrackCountingHighEffJetTags = ak7CalobTagger.NegativeTrackCountingHighEffJetTags
ak7CaloNegativeTrackCountingHighPur = ak7CalobTagger.NegativeTrackCountingHighPur
ak7CaloNegativeOnlyJetBProbabilityJetTags = ak7CalobTagger.NegativeOnlyJetBProbabilityJetTags
ak7CaloPositiveOnlyJetBProbabilityJetTags = ak7CalobTagger.PositiveOnlyJetBProbabilityJetTags

ak7CaloSecondaryVertexTagInfos = ak7CalobTagger.SecondaryVertexTagInfos
ak7CaloSimpleSecondaryVertexHighEffBJetTags = ak7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak7CaloSimpleSecondaryVertexHighPurBJetTags = ak7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak7CaloCombinedSecondaryVertexBJetTags = ak7CalobTagger.CombinedSecondaryVertexBJetTags
ak7CaloCombinedSecondaryVertexMVABJetTags = ak7CalobTagger.CombinedSecondaryVertexMVABJetTags

ak7CaloSecondaryVertexNegativeTagInfos = ak7CalobTagger.SecondaryVertexNegativeTagInfos
ak7CaloSimpleSecondaryVertexNegativeHighEffBJetTags = ak7CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
ak7CaloSimpleSecondaryVertexNegativeHighPurBJetTags = ak7CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
ak7CaloCombinedSecondaryVertexNegativeBJetTags = ak7CalobTagger.CombinedSecondaryVertexNegativeBJetTags
ak7CaloCombinedSecondaryVertexPositiveBJetTags = ak7CalobTagger.CombinedSecondaryVertexPositiveBJetTags

ak7CaloSoftMuonTagInfos = ak7CalobTagger.SoftMuonTagInfos
ak7CaloSoftMuonBJetTags = ak7CalobTagger.SoftMuonBJetTags
ak7CaloSoftMuonByIP3dBJetTags = ak7CalobTagger.SoftMuonByIP3dBJetTags
ak7CaloSoftMuonByPtBJetTags = ak7CalobTagger.SoftMuonByPtBJetTags
ak7CaloNegativeSoftMuonByPtBJetTags = ak7CalobTagger.NegativeSoftMuonByPtBJetTags
ak7CaloPositiveSoftMuonByPtBJetTags = ak7CalobTagger.PositiveSoftMuonByPtBJetTags

ak7CaloPatJetFlavourId = cms.Sequence(ak7CaloPatJetPartonAssociation*ak7CaloPatJetFlavourAssociation)

ak7CaloJetBtaggingIP       = cms.Sequence(ak7CaloImpactParameterTagInfos *
            (ak7CaloTrackCountingHighEffBJetTags +
             ak7CaloTrackCountingHighPurBJetTags +
             ak7CaloJetProbabilityBJetTags +
             ak7CaloJetBProbabilityBJetTags +
             ak7CaloPositiveOnlyJetProbabilityJetTags +
             ak7CaloNegativeOnlyJetProbabilityJetTags +
             ak7CaloNegativeTrackCountingHighEffJetTags +
             ak7CaloNegativeTrackCountingHighPur +
             ak7CaloNegativeOnlyJetBProbabilityJetTags +
             ak7CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

ak7CaloJetBtaggingSV = cms.Sequence(ak7CaloImpactParameterTagInfos
            *
            ak7CaloSecondaryVertexTagInfos
            * (ak7CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak7CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak7CaloCombinedSecondaryVertexBJetTags
                +
                ak7CaloCombinedSecondaryVertexMVABJetTags
              )
            )

ak7CaloJetBtaggingNegSV = cms.Sequence(ak7CaloImpactParameterTagInfos
            *
            ak7CaloSecondaryVertexNegativeTagInfos
            * (ak7CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                ak7CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                ak7CaloCombinedSecondaryVertexNegativeBJetTags
                +
                ak7CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

ak7CaloJetBtaggingMu = cms.Sequence(ak7CaloSoftMuonTagInfos * (ak7CaloSoftMuonBJetTags
                +
                ak7CaloSoftMuonByIP3dBJetTags
                +
                ak7CaloSoftMuonByPtBJetTags
                +
                ak7CaloNegativeSoftMuonByPtBJetTags
                +
                ak7CaloPositiveSoftMuonByPtBJetTags
              )
            )

ak7CaloJetBtagging = cms.Sequence(ak7CaloJetBtaggingIP
            *ak7CaloJetBtaggingSV
            *ak7CaloJetBtaggingNegSV
            *ak7CaloJetBtaggingMu
            )

ak7CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak7CaloJets"),
        genJetMatch          = cms.InputTag("ak7Calomatch"),
        genPartonMatch       = cms.InputTag("ak7Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak7Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak7CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak7CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("ak7CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak7CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak7CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak7CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("ak7CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak7CaloJetProbabilityBJetTags"),
            cms.InputTag("ak7CaloSoftMuonByPtBJetTags"),
            cms.InputTag("ak7CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("ak7CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak7CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak7CaloJetID"),
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

ak7CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak7CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
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
                                                             bTagJetName = cms.untracked.string("ak7Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak7CaloJetSequence_mc = cms.Sequence(
                                                  ak7Caloclean
                                                  *
                                                  ak7Calomatch
                                                  *
                                                  ak7Caloparton
                                                  *
                                                  ak7Calocorr
                                                  *
                                                  ak7CaloJetID
                                                  *
                                                  ak7CaloPatJetFlavourId
                                                  *
                                                  ak7CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak7CaloJetBtagging
                                                  *
                                                  ak7CalopatJetsWithBtagging
                                                  *
                                                  ak7CaloJetAnalyzer
                                                  )

ak7CaloJetSequence_data = cms.Sequence(ak7Calocorr
                                                    *
                                                    ak7CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak7CaloJetBtagging
                                                    *
                                                    ak7CalopatJetsWithBtagging
                                                    *
                                                    ak7CaloJetAnalyzer
                                                    )

ak7CaloJetSequence_jec = ak7CaloJetSequence_mc
ak7CaloJetSequence_mix = ak7CaloJetSequence_mc

ak7CaloJetSequence = cms.Sequence(ak7CaloJetSequence_mc)
