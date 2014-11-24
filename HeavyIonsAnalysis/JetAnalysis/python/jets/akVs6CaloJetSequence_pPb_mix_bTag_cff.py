

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs6CaloJets"),
    matched = cms.InputTag("ak6HiGenJets")
    )

akVs6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs6CaloJets")
                                                        )

akVs6Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs6CaloJets"),
    payload = "AKVs6Calo_HI"
    )

akVs6CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs6CaloJets'))

akVs6Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiGenJets'))

akVs6CalobTagger = bTaggers("akVs6Calo")

#create objects locally since they dont load properly otherwise
akVs6Calomatch = akVs6CalobTagger.match
akVs6Caloparton = akVs6CalobTagger.parton
akVs6CaloPatJetFlavourAssociation = akVs6CalobTagger.PatJetFlavourAssociation
akVs6CaloJetTracksAssociatorAtVertex = akVs6CalobTagger.JetTracksAssociatorAtVertex
akVs6CaloSimpleSecondaryVertexHighEffBJetTags = akVs6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs6CaloSimpleSecondaryVertexHighPurBJetTags = akVs6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs6CaloCombinedSecondaryVertexBJetTags = akVs6CalobTagger.CombinedSecondaryVertexBJetTags
akVs6CaloCombinedSecondaryVertexMVABJetTags = akVs6CalobTagger.CombinedSecondaryVertexMVABJetTags
akVs6CaloJetBProbabilityBJetTags = akVs6CalobTagger.JetBProbabilityBJetTags
akVs6CaloSoftMuonByPtBJetTags = akVs6CalobTagger.SoftMuonByPtBJetTags
akVs6CaloSoftMuonByIP3dBJetTags = akVs6CalobTagger.SoftMuonByIP3dBJetTags
akVs6CaloTrackCountingHighEffBJetTags = akVs6CalobTagger.TrackCountingHighEffBJetTags
akVs6CaloTrackCountingHighPurBJetTags = akVs6CalobTagger.TrackCountingHighPurBJetTags
akVs6CaloPatJetPartonAssociation = akVs6CalobTagger.PatJetPartonAssociation

akVs6CaloImpactParameterTagInfos = akVs6CalobTagger.ImpactParameterTagInfos
akVs6CaloJetProbabilityBJetTags = akVs6CalobTagger.JetProbabilityBJetTags
akVs6CaloPositiveOnlyJetProbabilityJetTags = akVs6CalobTagger.PositiveOnlyJetProbabilityJetTags
akVs6CaloNegativeOnlyJetProbabilityJetTags = akVs6CalobTagger.NegativeOnlyJetProbabilityJetTags
akVs6CaloNegativeTrackCountingHighEffJetTags = akVs6CalobTagger.NegativeTrackCountingHighEffJetTags
akVs6CaloNegativeTrackCountingHighPur = akVs6CalobTagger.NegativeTrackCountingHighPur
akVs6CaloNegativeOnlyJetBProbabilityJetTags = akVs6CalobTagger.NegativeOnlyJetBProbabilityJetTags
akVs6CaloPositiveOnlyJetBProbabilityJetTags = akVs6CalobTagger.PositiveOnlyJetBProbabilityJetTags

akVs6CaloSecondaryVertexTagInfos = akVs6CalobTagger.SecondaryVertexTagInfos
akVs6CaloSimpleSecondaryVertexHighEffBJetTags = akVs6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs6CaloSimpleSecondaryVertexHighPurBJetTags = akVs6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs6CaloCombinedSecondaryVertexBJetTags = akVs6CalobTagger.CombinedSecondaryVertexBJetTags
akVs6CaloCombinedSecondaryVertexMVABJetTags = akVs6CalobTagger.CombinedSecondaryVertexMVABJetTags

akVs6CaloSecondaryVertexNegativeTagInfos = akVs6CalobTagger.SecondaryVertexNegativeTagInfos
akVs6CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akVs6CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs6CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akVs6CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs6CaloCombinedSecondaryVertexNegativeBJetTags = akVs6CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akVs6CaloCombinedSecondaryVertexPositiveBJetTags = akVs6CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akVs6CaloSoftMuonTagInfos = akVs6CalobTagger.SoftMuonTagInfos
akVs6CaloSoftMuonBJetTags = akVs6CalobTagger.SoftMuonBJetTags
akVs6CaloSoftMuonByIP3dBJetTags = akVs6CalobTagger.SoftMuonByIP3dBJetTags
akVs6CaloSoftMuonByPtBJetTags = akVs6CalobTagger.SoftMuonByPtBJetTags
akVs6CaloNegativeSoftMuonByPtBJetTags = akVs6CalobTagger.NegativeSoftMuonByPtBJetTags
akVs6CaloPositiveSoftMuonByPtBJetTags = akVs6CalobTagger.PositiveSoftMuonByPtBJetTags

akVs6CaloPatJetFlavourId = cms.Sequence(akVs6CaloPatJetPartonAssociation*akVs6CaloPatJetFlavourAssociation)

akVs6CaloJetBtaggingIP       = cms.Sequence(akVs6CaloImpactParameterTagInfos *
            (akVs6CaloTrackCountingHighEffBJetTags +
             akVs6CaloTrackCountingHighPurBJetTags +
             akVs6CaloJetProbabilityBJetTags +
             akVs6CaloJetBProbabilityBJetTags +
             akVs6CaloPositiveOnlyJetProbabilityJetTags +
             akVs6CaloNegativeOnlyJetProbabilityJetTags +
             akVs6CaloNegativeTrackCountingHighEffJetTags +
             akVs6CaloNegativeTrackCountingHighPur +
             akVs6CaloNegativeOnlyJetBProbabilityJetTags +
             akVs6CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs6CaloJetBtaggingSV = cms.Sequence(akVs6CaloImpactParameterTagInfos
            *
            akVs6CaloSecondaryVertexTagInfos
            * (akVs6CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs6CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs6CaloCombinedSecondaryVertexBJetTags
                +
                akVs6CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akVs6CaloJetBtaggingNegSV = cms.Sequence(akVs6CaloImpactParameterTagInfos
            *
            akVs6CaloSecondaryVertexNegativeTagInfos
            * (akVs6CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs6CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs6CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akVs6CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs6CaloJetBtaggingMu = cms.Sequence(akVs6CaloSoftMuonTagInfos * (akVs6CaloSoftMuonBJetTags
                +
                akVs6CaloSoftMuonByIP3dBJetTags
                +
                akVs6CaloSoftMuonByPtBJetTags
                +
                akVs6CaloNegativeSoftMuonByPtBJetTags
                +
                akVs6CaloPositiveSoftMuonByPtBJetTags
              )
            )

akVs6CaloJetBtagging = cms.Sequence(akVs6CaloJetBtaggingIP
            *akVs6CaloJetBtaggingSV
            *akVs6CaloJetBtaggingNegSV
            *akVs6CaloJetBtaggingMu
            )

akVs6CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs6CaloJets"),
        genJetMatch          = cms.InputTag("akVs6Calomatch"),
        genPartonMatch       = cms.InputTag("akVs6Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs6Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs6CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs6CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs6CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs6CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs6CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs6CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs6CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs6CaloJetProbabilityBJetTags"),
            cms.InputTag("akVs6CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akVs6CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs6CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs6CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs6CaloJetID"),
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

akVs6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs6CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("hiSignal"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs6Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs6CaloJetSequence_mc = cms.Sequence(
                                                  akVs6Caloclean
                                                  *
                                                  akVs6Calomatch
                                                  *
                                                  akVs6Caloparton
                                                  *
                                                  akVs6Calocorr
                                                  *
                                                  akVs6CaloJetID
                                                  *
                                                  akVs6CaloPatJetFlavourId
                                                  *
                                                  akVs6CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs6CaloJetBtagging
                                                  *
                                                  akVs6CalopatJetsWithBtagging
                                                  *
                                                  akVs6CaloJetAnalyzer
                                                  )

akVs6CaloJetSequence_data = cms.Sequence(akVs6Calocorr
                                                    *
                                                    akVs6CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs6CaloJetBtagging
                                                    *
                                                    akVs6CalopatJetsWithBtagging
                                                    *
                                                    akVs6CaloJetAnalyzer
                                                    )

akVs6CaloJetSequence_jec = akVs6CaloJetSequence_mc
akVs6CaloJetSequence_mix = akVs6CaloJetSequence_mc

akVs6CaloJetSequence = cms.Sequence(akVs6CaloJetSequence_mix)
