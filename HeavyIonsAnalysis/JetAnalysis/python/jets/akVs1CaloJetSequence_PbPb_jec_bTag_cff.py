

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs1CaloJets"),
    matched = cms.InputTag("ak1HiGenJetsCleaned")
    )

akVs1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs1CaloJets")
                                                        )

akVs1Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs1CaloJets"),
    payload = "AKVs1Calo_HI"
    )

akVs1CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs1CaloJets'))

akVs1Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiGenJetsCleaned'))

akVs1CalobTagger = bTaggers("akVs1Calo")

#create objects locally since they dont load properly otherwise
akVs1Calomatch = akVs1CalobTagger.match
akVs1Caloparton = akVs1CalobTagger.parton
akVs1CaloPatJetFlavourAssociation = akVs1CalobTagger.PatJetFlavourAssociation
akVs1CaloJetTracksAssociatorAtVertex = akVs1CalobTagger.JetTracksAssociatorAtVertex
akVs1CaloSimpleSecondaryVertexHighEffBJetTags = akVs1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs1CaloSimpleSecondaryVertexHighPurBJetTags = akVs1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs1CaloCombinedSecondaryVertexBJetTags = akVs1CalobTagger.CombinedSecondaryVertexBJetTags
akVs1CaloCombinedSecondaryVertexMVABJetTags = akVs1CalobTagger.CombinedSecondaryVertexMVABJetTags
akVs1CaloJetBProbabilityBJetTags = akVs1CalobTagger.JetBProbabilityBJetTags
akVs1CaloSoftMuonByPtBJetTags = akVs1CalobTagger.SoftMuonByPtBJetTags
akVs1CaloSoftMuonByIP3dBJetTags = akVs1CalobTagger.SoftMuonByIP3dBJetTags
akVs1CaloTrackCountingHighEffBJetTags = akVs1CalobTagger.TrackCountingHighEffBJetTags
akVs1CaloTrackCountingHighPurBJetTags = akVs1CalobTagger.TrackCountingHighPurBJetTags
akVs1CaloPatJetPartonAssociation = akVs1CalobTagger.PatJetPartonAssociation

akVs1CaloImpactParameterTagInfos = akVs1CalobTagger.ImpactParameterTagInfos
akVs1CaloJetProbabilityBJetTags = akVs1CalobTagger.JetProbabilityBJetTags
akVs1CaloPositiveOnlyJetProbabilityJetTags = akVs1CalobTagger.PositiveOnlyJetProbabilityJetTags
akVs1CaloNegativeOnlyJetProbabilityJetTags = akVs1CalobTagger.NegativeOnlyJetProbabilityJetTags
akVs1CaloNegativeTrackCountingHighEffJetTags = akVs1CalobTagger.NegativeTrackCountingHighEffJetTags
akVs1CaloNegativeTrackCountingHighPur = akVs1CalobTagger.NegativeTrackCountingHighPur
akVs1CaloNegativeOnlyJetBProbabilityJetTags = akVs1CalobTagger.NegativeOnlyJetBProbabilityJetTags
akVs1CaloPositiveOnlyJetBProbabilityJetTags = akVs1CalobTagger.PositiveOnlyJetBProbabilityJetTags

akVs1CaloSecondaryVertexTagInfos = akVs1CalobTagger.SecondaryVertexTagInfos
akVs1CaloSimpleSecondaryVertexHighEffBJetTags = akVs1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs1CaloSimpleSecondaryVertexHighPurBJetTags = akVs1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs1CaloCombinedSecondaryVertexBJetTags = akVs1CalobTagger.CombinedSecondaryVertexBJetTags
akVs1CaloCombinedSecondaryVertexMVABJetTags = akVs1CalobTagger.CombinedSecondaryVertexMVABJetTags

akVs1CaloSecondaryVertexNegativeTagInfos = akVs1CalobTagger.SecondaryVertexNegativeTagInfos
akVs1CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akVs1CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs1CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akVs1CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs1CaloCombinedSecondaryVertexNegativeBJetTags = akVs1CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akVs1CaloCombinedSecondaryVertexPositiveBJetTags = akVs1CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akVs1CaloSoftMuonTagInfos = akVs1CalobTagger.SoftMuonTagInfos
akVs1CaloSoftMuonBJetTags = akVs1CalobTagger.SoftMuonBJetTags
akVs1CaloSoftMuonByIP3dBJetTags = akVs1CalobTagger.SoftMuonByIP3dBJetTags
akVs1CaloSoftMuonByPtBJetTags = akVs1CalobTagger.SoftMuonByPtBJetTags
akVs1CaloNegativeSoftMuonByPtBJetTags = akVs1CalobTagger.NegativeSoftMuonByPtBJetTags
akVs1CaloPositiveSoftMuonByPtBJetTags = akVs1CalobTagger.PositiveSoftMuonByPtBJetTags

akVs1CaloPatJetFlavourId = cms.Sequence(akVs1CaloPatJetPartonAssociation*akVs1CaloPatJetFlavourAssociation)

akVs1CaloJetBtaggingIP       = cms.Sequence(akVs1CaloImpactParameterTagInfos *
            (akVs1CaloTrackCountingHighEffBJetTags +
             akVs1CaloTrackCountingHighPurBJetTags +
             akVs1CaloJetProbabilityBJetTags +
             akVs1CaloJetBProbabilityBJetTags +
             akVs1CaloPositiveOnlyJetProbabilityJetTags +
             akVs1CaloNegativeOnlyJetProbabilityJetTags +
             akVs1CaloNegativeTrackCountingHighEffJetTags +
             akVs1CaloNegativeTrackCountingHighPur +
             akVs1CaloNegativeOnlyJetBProbabilityJetTags +
             akVs1CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs1CaloJetBtaggingSV = cms.Sequence(akVs1CaloImpactParameterTagInfos
            *
            akVs1CaloSecondaryVertexTagInfos
            * (akVs1CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs1CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs1CaloCombinedSecondaryVertexBJetTags
                +
                akVs1CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akVs1CaloJetBtaggingNegSV = cms.Sequence(akVs1CaloImpactParameterTagInfos
            *
            akVs1CaloSecondaryVertexNegativeTagInfos
            * (akVs1CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs1CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs1CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akVs1CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs1CaloJetBtaggingMu = cms.Sequence(akVs1CaloSoftMuonTagInfos * (akVs1CaloSoftMuonBJetTags
                +
                akVs1CaloSoftMuonByIP3dBJetTags
                +
                akVs1CaloSoftMuonByPtBJetTags
                +
                akVs1CaloNegativeSoftMuonByPtBJetTags
                +
                akVs1CaloPositiveSoftMuonByPtBJetTags
              )
            )

akVs1CaloJetBtagging = cms.Sequence(akVs1CaloJetBtaggingIP
            *akVs1CaloJetBtaggingSV
            *akVs1CaloJetBtaggingNegSV
            *akVs1CaloJetBtaggingMu
            )

akVs1CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs1CaloJets"),
        genJetMatch          = cms.InputTag("akVs1Calomatch"),
        genPartonMatch       = cms.InputTag("akVs1Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs1Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs1CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs1CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs1CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs1CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs1CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs1CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs1CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs1CaloJetProbabilityBJetTags"),
            cms.InputTag("akVs1CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akVs1CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs1CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs1CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs1CaloJetID"),
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

akVs1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs1CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJetsCleaned',
                                                             rParam = 0.1,
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
                                                             bTagJetName = cms.untracked.string("akVs1Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs1CaloJetSequence_mc = cms.Sequence(
                                                  akVs1Caloclean
                                                  *
                                                  akVs1Calomatch
                                                  *
                                                  akVs1Caloparton
                                                  *
                                                  akVs1Calocorr
                                                  *
                                                  akVs1CaloJetID
                                                  *
                                                  akVs1CaloPatJetFlavourId
                                                  *
                                                  akVs1CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs1CaloJetBtagging
                                                  *
                                                  akVs1CalopatJetsWithBtagging
                                                  *
                                                  akVs1CaloJetAnalyzer
                                                  )

akVs1CaloJetSequence_data = cms.Sequence(akVs1Calocorr
                                                    *
                                                    akVs1CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs1CaloJetBtagging
                                                    *
                                                    akVs1CalopatJetsWithBtagging
                                                    *
                                                    akVs1CaloJetAnalyzer
                                                    )

akVs1CaloJetSequence_jec = akVs1CaloJetSequence_mc
akVs1CaloJetSequence_mix = akVs1CaloJetSequence_mc

akVs1CaloJetSequence = cms.Sequence(akVs1CaloJetSequence_jec)
akVs1CaloJetAnalyzer.genPtMin = cms.untracked.double(1)
