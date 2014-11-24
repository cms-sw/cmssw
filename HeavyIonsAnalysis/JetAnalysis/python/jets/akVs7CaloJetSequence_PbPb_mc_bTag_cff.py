

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs7Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs7CaloJets"),
    matched = cms.InputTag("ak7HiGenJetsCleaned")
    )

akVs7Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs7CaloJets")
                                                        )

akVs7Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs7CaloJets"),
    payload = "AKVs7Calo_HI"
    )

akVs7CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs7CaloJets'))

akVs7Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJetsCleaned'))

akVs7CalobTagger = bTaggers("akVs7Calo")

#create objects locally since they dont load properly otherwise
akVs7Calomatch = akVs7CalobTagger.match
akVs7Caloparton = akVs7CalobTagger.parton
akVs7CaloPatJetFlavourAssociation = akVs7CalobTagger.PatJetFlavourAssociation
akVs7CaloJetTracksAssociatorAtVertex = akVs7CalobTagger.JetTracksAssociatorAtVertex
akVs7CaloSimpleSecondaryVertexHighEffBJetTags = akVs7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs7CaloSimpleSecondaryVertexHighPurBJetTags = akVs7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs7CaloCombinedSecondaryVertexBJetTags = akVs7CalobTagger.CombinedSecondaryVertexBJetTags
akVs7CaloCombinedSecondaryVertexMVABJetTags = akVs7CalobTagger.CombinedSecondaryVertexMVABJetTags
akVs7CaloJetBProbabilityBJetTags = akVs7CalobTagger.JetBProbabilityBJetTags
akVs7CaloSoftMuonByPtBJetTags = akVs7CalobTagger.SoftMuonByPtBJetTags
akVs7CaloSoftMuonByIP3dBJetTags = akVs7CalobTagger.SoftMuonByIP3dBJetTags
akVs7CaloTrackCountingHighEffBJetTags = akVs7CalobTagger.TrackCountingHighEffBJetTags
akVs7CaloTrackCountingHighPurBJetTags = akVs7CalobTagger.TrackCountingHighPurBJetTags
akVs7CaloPatJetPartonAssociation = akVs7CalobTagger.PatJetPartonAssociation

akVs7CaloImpactParameterTagInfos = akVs7CalobTagger.ImpactParameterTagInfos
akVs7CaloJetProbabilityBJetTags = akVs7CalobTagger.JetProbabilityBJetTags
akVs7CaloPositiveOnlyJetProbabilityJetTags = akVs7CalobTagger.PositiveOnlyJetProbabilityJetTags
akVs7CaloNegativeOnlyJetProbabilityJetTags = akVs7CalobTagger.NegativeOnlyJetProbabilityJetTags
akVs7CaloNegativeTrackCountingHighEffJetTags = akVs7CalobTagger.NegativeTrackCountingHighEffJetTags
akVs7CaloNegativeTrackCountingHighPur = akVs7CalobTagger.NegativeTrackCountingHighPur
akVs7CaloNegativeOnlyJetBProbabilityJetTags = akVs7CalobTagger.NegativeOnlyJetBProbabilityJetTags
akVs7CaloPositiveOnlyJetBProbabilityJetTags = akVs7CalobTagger.PositiveOnlyJetBProbabilityJetTags

akVs7CaloSecondaryVertexTagInfos = akVs7CalobTagger.SecondaryVertexTagInfos
akVs7CaloSimpleSecondaryVertexHighEffBJetTags = akVs7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs7CaloSimpleSecondaryVertexHighPurBJetTags = akVs7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs7CaloCombinedSecondaryVertexBJetTags = akVs7CalobTagger.CombinedSecondaryVertexBJetTags
akVs7CaloCombinedSecondaryVertexMVABJetTags = akVs7CalobTagger.CombinedSecondaryVertexMVABJetTags

akVs7CaloSecondaryVertexNegativeTagInfos = akVs7CalobTagger.SecondaryVertexNegativeTagInfos
akVs7CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akVs7CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs7CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akVs7CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs7CaloCombinedSecondaryVertexNegativeBJetTags = akVs7CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akVs7CaloCombinedSecondaryVertexPositiveBJetTags = akVs7CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akVs7CaloSoftMuonTagInfos = akVs7CalobTagger.SoftMuonTagInfos
akVs7CaloSoftMuonBJetTags = akVs7CalobTagger.SoftMuonBJetTags
akVs7CaloSoftMuonByIP3dBJetTags = akVs7CalobTagger.SoftMuonByIP3dBJetTags
akVs7CaloSoftMuonByPtBJetTags = akVs7CalobTagger.SoftMuonByPtBJetTags
akVs7CaloNegativeSoftMuonByPtBJetTags = akVs7CalobTagger.NegativeSoftMuonByPtBJetTags
akVs7CaloPositiveSoftMuonByPtBJetTags = akVs7CalobTagger.PositiveSoftMuonByPtBJetTags

akVs7CaloPatJetFlavourId = cms.Sequence(akVs7CaloPatJetPartonAssociation*akVs7CaloPatJetFlavourAssociation)

akVs7CaloJetBtaggingIP       = cms.Sequence(akVs7CaloImpactParameterTagInfos *
            (akVs7CaloTrackCountingHighEffBJetTags +
             akVs7CaloTrackCountingHighPurBJetTags +
             akVs7CaloJetProbabilityBJetTags +
             akVs7CaloJetBProbabilityBJetTags +
             akVs7CaloPositiveOnlyJetProbabilityJetTags +
             akVs7CaloNegativeOnlyJetProbabilityJetTags +
             akVs7CaloNegativeTrackCountingHighEffJetTags +
             akVs7CaloNegativeTrackCountingHighPur +
             akVs7CaloNegativeOnlyJetBProbabilityJetTags +
             akVs7CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs7CaloJetBtaggingSV = cms.Sequence(akVs7CaloImpactParameterTagInfos
            *
            akVs7CaloSecondaryVertexTagInfos
            * (akVs7CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs7CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs7CaloCombinedSecondaryVertexBJetTags
                +
                akVs7CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akVs7CaloJetBtaggingNegSV = cms.Sequence(akVs7CaloImpactParameterTagInfos
            *
            akVs7CaloSecondaryVertexNegativeTagInfos
            * (akVs7CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs7CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs7CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akVs7CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs7CaloJetBtaggingMu = cms.Sequence(akVs7CaloSoftMuonTagInfos * (akVs7CaloSoftMuonBJetTags
                +
                akVs7CaloSoftMuonByIP3dBJetTags
                +
                akVs7CaloSoftMuonByPtBJetTags
                +
                akVs7CaloNegativeSoftMuonByPtBJetTags
                +
                akVs7CaloPositiveSoftMuonByPtBJetTags
              )
            )

akVs7CaloJetBtagging = cms.Sequence(akVs7CaloJetBtaggingIP
            *akVs7CaloJetBtaggingSV
            *akVs7CaloJetBtaggingNegSV
            *akVs7CaloJetBtaggingMu
            )

akVs7CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs7CaloJets"),
        genJetMatch          = cms.InputTag("akVs7Calomatch"),
        genPartonMatch       = cms.InputTag("akVs7Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs7Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs7CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs7CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs7CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs7CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs7CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs7CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs7CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs7CaloJetProbabilityBJetTags"),
            cms.InputTag("akVs7CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akVs7CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs7CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs7CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs7CaloJetID"),
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

akVs7CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs7CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJetsCleaned',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs7Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs7CaloJetSequence_mc = cms.Sequence(
                                                  akVs7Caloclean
                                                  *
                                                  akVs7Calomatch
                                                  *
                                                  akVs7Caloparton
                                                  *
                                                  akVs7Calocorr
                                                  *
                                                  akVs7CaloJetID
                                                  *
                                                  akVs7CaloPatJetFlavourId
                                                  *
                                                  akVs7CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs7CaloJetBtagging
                                                  *
                                                  akVs7CalopatJetsWithBtagging
                                                  *
                                                  akVs7CaloJetAnalyzer
                                                  )

akVs7CaloJetSequence_data = cms.Sequence(akVs7Calocorr
                                                    *
                                                    akVs7CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs7CaloJetBtagging
                                                    *
                                                    akVs7CalopatJetsWithBtagging
                                                    *
                                                    akVs7CaloJetAnalyzer
                                                    )

akVs7CaloJetSequence_jec = akVs7CaloJetSequence_mc
akVs7CaloJetSequence_mix = akVs7CaloJetSequence_mc

akVs7CaloJetSequence = cms.Sequence(akVs7CaloJetSequence_mc)
