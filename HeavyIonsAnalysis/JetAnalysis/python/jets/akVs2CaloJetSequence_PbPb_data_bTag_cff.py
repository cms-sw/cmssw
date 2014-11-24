

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs2CaloJets"),
    matched = cms.InputTag("ak2HiGenJetsCleaned")
    )

akVs2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs2CaloJets")
                                                        )

akVs2Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs2CaloJets"),
    payload = "AKVs2Calo_HI"
    )

akVs2CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs2CaloJets'))

akVs2Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJetsCleaned'))

akVs2CalobTagger = bTaggers("akVs2Calo")

#create objects locally since they dont load properly otherwise
akVs2Calomatch = akVs2CalobTagger.match
akVs2Caloparton = akVs2CalobTagger.parton
akVs2CaloPatJetFlavourAssociation = akVs2CalobTagger.PatJetFlavourAssociation
akVs2CaloJetTracksAssociatorAtVertex = akVs2CalobTagger.JetTracksAssociatorAtVertex
akVs2CaloSimpleSecondaryVertexHighEffBJetTags = akVs2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs2CaloSimpleSecondaryVertexHighPurBJetTags = akVs2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs2CaloCombinedSecondaryVertexBJetTags = akVs2CalobTagger.CombinedSecondaryVertexBJetTags
akVs2CaloCombinedSecondaryVertexMVABJetTags = akVs2CalobTagger.CombinedSecondaryVertexMVABJetTags
akVs2CaloJetBProbabilityBJetTags = akVs2CalobTagger.JetBProbabilityBJetTags
akVs2CaloSoftMuonByPtBJetTags = akVs2CalobTagger.SoftMuonByPtBJetTags
akVs2CaloSoftMuonByIP3dBJetTags = akVs2CalobTagger.SoftMuonByIP3dBJetTags
akVs2CaloTrackCountingHighEffBJetTags = akVs2CalobTagger.TrackCountingHighEffBJetTags
akVs2CaloTrackCountingHighPurBJetTags = akVs2CalobTagger.TrackCountingHighPurBJetTags
akVs2CaloPatJetPartonAssociation = akVs2CalobTagger.PatJetPartonAssociation

akVs2CaloImpactParameterTagInfos = akVs2CalobTagger.ImpactParameterTagInfos
akVs2CaloJetProbabilityBJetTags = akVs2CalobTagger.JetProbabilityBJetTags
akVs2CaloPositiveOnlyJetProbabilityJetTags = akVs2CalobTagger.PositiveOnlyJetProbabilityJetTags
akVs2CaloNegativeOnlyJetProbabilityJetTags = akVs2CalobTagger.NegativeOnlyJetProbabilityJetTags
akVs2CaloNegativeTrackCountingHighEffJetTags = akVs2CalobTagger.NegativeTrackCountingHighEffJetTags
akVs2CaloNegativeTrackCountingHighPur = akVs2CalobTagger.NegativeTrackCountingHighPur
akVs2CaloNegativeOnlyJetBProbabilityJetTags = akVs2CalobTagger.NegativeOnlyJetBProbabilityJetTags
akVs2CaloPositiveOnlyJetBProbabilityJetTags = akVs2CalobTagger.PositiveOnlyJetBProbabilityJetTags

akVs2CaloSecondaryVertexTagInfos = akVs2CalobTagger.SecondaryVertexTagInfos
akVs2CaloSimpleSecondaryVertexHighEffBJetTags = akVs2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs2CaloSimpleSecondaryVertexHighPurBJetTags = akVs2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs2CaloCombinedSecondaryVertexBJetTags = akVs2CalobTagger.CombinedSecondaryVertexBJetTags
akVs2CaloCombinedSecondaryVertexMVABJetTags = akVs2CalobTagger.CombinedSecondaryVertexMVABJetTags

akVs2CaloSecondaryVertexNegativeTagInfos = akVs2CalobTagger.SecondaryVertexNegativeTagInfos
akVs2CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akVs2CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs2CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akVs2CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs2CaloCombinedSecondaryVertexNegativeBJetTags = akVs2CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akVs2CaloCombinedSecondaryVertexPositiveBJetTags = akVs2CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akVs2CaloSoftMuonTagInfos = akVs2CalobTagger.SoftMuonTagInfos
akVs2CaloSoftMuonBJetTags = akVs2CalobTagger.SoftMuonBJetTags
akVs2CaloSoftMuonByIP3dBJetTags = akVs2CalobTagger.SoftMuonByIP3dBJetTags
akVs2CaloSoftMuonByPtBJetTags = akVs2CalobTagger.SoftMuonByPtBJetTags
akVs2CaloNegativeSoftMuonByPtBJetTags = akVs2CalobTagger.NegativeSoftMuonByPtBJetTags
akVs2CaloPositiveSoftMuonByPtBJetTags = akVs2CalobTagger.PositiveSoftMuonByPtBJetTags

akVs2CaloPatJetFlavourId = cms.Sequence(akVs2CaloPatJetPartonAssociation*akVs2CaloPatJetFlavourAssociation)

akVs2CaloJetBtaggingIP       = cms.Sequence(akVs2CaloImpactParameterTagInfos *
            (akVs2CaloTrackCountingHighEffBJetTags +
             akVs2CaloTrackCountingHighPurBJetTags +
             akVs2CaloJetProbabilityBJetTags +
             akVs2CaloJetBProbabilityBJetTags +
             akVs2CaloPositiveOnlyJetProbabilityJetTags +
             akVs2CaloNegativeOnlyJetProbabilityJetTags +
             akVs2CaloNegativeTrackCountingHighEffJetTags +
             akVs2CaloNegativeTrackCountingHighPur +
             akVs2CaloNegativeOnlyJetBProbabilityJetTags +
             akVs2CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs2CaloJetBtaggingSV = cms.Sequence(akVs2CaloImpactParameterTagInfos
            *
            akVs2CaloSecondaryVertexTagInfos
            * (akVs2CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs2CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs2CaloCombinedSecondaryVertexBJetTags
                +
                akVs2CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akVs2CaloJetBtaggingNegSV = cms.Sequence(akVs2CaloImpactParameterTagInfos
            *
            akVs2CaloSecondaryVertexNegativeTagInfos
            * (akVs2CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs2CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs2CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akVs2CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs2CaloJetBtaggingMu = cms.Sequence(akVs2CaloSoftMuonTagInfos * (akVs2CaloSoftMuonBJetTags
                +
                akVs2CaloSoftMuonByIP3dBJetTags
                +
                akVs2CaloSoftMuonByPtBJetTags
                +
                akVs2CaloNegativeSoftMuonByPtBJetTags
                +
                akVs2CaloPositiveSoftMuonByPtBJetTags
              )
            )

akVs2CaloJetBtagging = cms.Sequence(akVs2CaloJetBtaggingIP
            *akVs2CaloJetBtaggingSV
            *akVs2CaloJetBtaggingNegSV
            *akVs2CaloJetBtaggingMu
            )

akVs2CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs2CaloJets"),
        genJetMatch          = cms.InputTag("akVs2Calomatch"),
        genPartonMatch       = cms.InputTag("akVs2Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs2Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs2CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs2CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs2CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs2CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs2CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs2CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs2CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs2CaloJetProbabilityBJetTags"),
            cms.InputTag("akVs2CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akVs2CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs2CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs2CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs2CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = True,
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        embedCaloTowers = False,
        embedPFCandidates = True
        )

akVs2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs2CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak2HiGenJetsCleaned',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs2Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs2CaloJetSequence_mc = cms.Sequence(
                                                  akVs2Caloclean
                                                  *
                                                  akVs2Calomatch
                                                  *
                                                  akVs2Caloparton
                                                  *
                                                  akVs2Calocorr
                                                  *
                                                  akVs2CaloJetID
                                                  *
                                                  akVs2CaloPatJetFlavourId
                                                  *
                                                  akVs2CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs2CaloJetBtagging
                                                  *
                                                  akVs2CalopatJetsWithBtagging
                                                  *
                                                  akVs2CaloJetAnalyzer
                                                  )

akVs2CaloJetSequence_data = cms.Sequence(akVs2Calocorr
                                                    *
                                                    akVs2CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs2CaloJetBtagging
                                                    *
                                                    akVs2CalopatJetsWithBtagging
                                                    *
                                                    akVs2CaloJetAnalyzer
                                                    )

akVs2CaloJetSequence_jec = akVs2CaloJetSequence_mc
akVs2CaloJetSequence_mix = akVs2CaloJetSequence_mc

akVs2CaloJetSequence = cms.Sequence(akVs2CaloJetSequence_data)
