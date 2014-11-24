

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs5CaloJets"),
    matched = cms.InputTag("ak5HiGenJets")
    )

akVs5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs5CaloJets")
                                                        )

akVs5Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs5CaloJets"),
    payload = "AKVs5Calo_HI"
    )

akVs5CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs5CaloJets'))

akVs5Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiGenJets'))

akVs5CalobTagger = bTaggers("akVs5Calo")

#create objects locally since they dont load properly otherwise
akVs5Calomatch = akVs5CalobTagger.match
akVs5Caloparton = akVs5CalobTagger.parton
akVs5CaloPatJetFlavourAssociation = akVs5CalobTagger.PatJetFlavourAssociation
akVs5CaloJetTracksAssociatorAtVertex = akVs5CalobTagger.JetTracksAssociatorAtVertex
akVs5CaloSimpleSecondaryVertexHighEffBJetTags = akVs5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs5CaloSimpleSecondaryVertexHighPurBJetTags = akVs5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs5CaloCombinedSecondaryVertexBJetTags = akVs5CalobTagger.CombinedSecondaryVertexBJetTags
akVs5CaloCombinedSecondaryVertexMVABJetTags = akVs5CalobTagger.CombinedSecondaryVertexMVABJetTags
akVs5CaloJetBProbabilityBJetTags = akVs5CalobTagger.JetBProbabilityBJetTags
akVs5CaloSoftMuonByPtBJetTags = akVs5CalobTagger.SoftMuonByPtBJetTags
akVs5CaloSoftMuonByIP3dBJetTags = akVs5CalobTagger.SoftMuonByIP3dBJetTags
akVs5CaloTrackCountingHighEffBJetTags = akVs5CalobTagger.TrackCountingHighEffBJetTags
akVs5CaloTrackCountingHighPurBJetTags = akVs5CalobTagger.TrackCountingHighPurBJetTags
akVs5CaloPatJetPartonAssociation = akVs5CalobTagger.PatJetPartonAssociation

akVs5CaloImpactParameterTagInfos = akVs5CalobTagger.ImpactParameterTagInfos
akVs5CaloJetProbabilityBJetTags = akVs5CalobTagger.JetProbabilityBJetTags
akVs5CaloPositiveOnlyJetProbabilityJetTags = akVs5CalobTagger.PositiveOnlyJetProbabilityJetTags
akVs5CaloNegativeOnlyJetProbabilityJetTags = akVs5CalobTagger.NegativeOnlyJetProbabilityJetTags
akVs5CaloNegativeTrackCountingHighEffJetTags = akVs5CalobTagger.NegativeTrackCountingHighEffJetTags
akVs5CaloNegativeTrackCountingHighPur = akVs5CalobTagger.NegativeTrackCountingHighPur
akVs5CaloNegativeOnlyJetBProbabilityJetTags = akVs5CalobTagger.NegativeOnlyJetBProbabilityJetTags
akVs5CaloPositiveOnlyJetBProbabilityJetTags = akVs5CalobTagger.PositiveOnlyJetBProbabilityJetTags

akVs5CaloSecondaryVertexTagInfos = akVs5CalobTagger.SecondaryVertexTagInfos
akVs5CaloSimpleSecondaryVertexHighEffBJetTags = akVs5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs5CaloSimpleSecondaryVertexHighPurBJetTags = akVs5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs5CaloCombinedSecondaryVertexBJetTags = akVs5CalobTagger.CombinedSecondaryVertexBJetTags
akVs5CaloCombinedSecondaryVertexMVABJetTags = akVs5CalobTagger.CombinedSecondaryVertexMVABJetTags

akVs5CaloSecondaryVertexNegativeTagInfos = akVs5CalobTagger.SecondaryVertexNegativeTagInfos
akVs5CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akVs5CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs5CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akVs5CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs5CaloCombinedSecondaryVertexNegativeBJetTags = akVs5CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akVs5CaloCombinedSecondaryVertexPositiveBJetTags = akVs5CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akVs5CaloSoftMuonTagInfos = akVs5CalobTagger.SoftMuonTagInfos
akVs5CaloSoftMuonBJetTags = akVs5CalobTagger.SoftMuonBJetTags
akVs5CaloSoftMuonByIP3dBJetTags = akVs5CalobTagger.SoftMuonByIP3dBJetTags
akVs5CaloSoftMuonByPtBJetTags = akVs5CalobTagger.SoftMuonByPtBJetTags
akVs5CaloNegativeSoftMuonByPtBJetTags = akVs5CalobTagger.NegativeSoftMuonByPtBJetTags
akVs5CaloPositiveSoftMuonByPtBJetTags = akVs5CalobTagger.PositiveSoftMuonByPtBJetTags

akVs5CaloPatJetFlavourId = cms.Sequence(akVs5CaloPatJetPartonAssociation*akVs5CaloPatJetFlavourAssociation)

akVs5CaloJetBtaggingIP       = cms.Sequence(akVs5CaloImpactParameterTagInfos *
            (akVs5CaloTrackCountingHighEffBJetTags +
             akVs5CaloTrackCountingHighPurBJetTags +
             akVs5CaloJetProbabilityBJetTags +
             akVs5CaloJetBProbabilityBJetTags +
             akVs5CaloPositiveOnlyJetProbabilityJetTags +
             akVs5CaloNegativeOnlyJetProbabilityJetTags +
             akVs5CaloNegativeTrackCountingHighEffJetTags +
             akVs5CaloNegativeTrackCountingHighPur +
             akVs5CaloNegativeOnlyJetBProbabilityJetTags +
             akVs5CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs5CaloJetBtaggingSV = cms.Sequence(akVs5CaloImpactParameterTagInfos
            *
            akVs5CaloSecondaryVertexTagInfos
            * (akVs5CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs5CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs5CaloCombinedSecondaryVertexBJetTags
                +
                akVs5CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akVs5CaloJetBtaggingNegSV = cms.Sequence(akVs5CaloImpactParameterTagInfos
            *
            akVs5CaloSecondaryVertexNegativeTagInfos
            * (akVs5CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs5CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs5CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akVs5CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs5CaloJetBtaggingMu = cms.Sequence(akVs5CaloSoftMuonTagInfos * (akVs5CaloSoftMuonBJetTags
                +
                akVs5CaloSoftMuonByIP3dBJetTags
                +
                akVs5CaloSoftMuonByPtBJetTags
                +
                akVs5CaloNegativeSoftMuonByPtBJetTags
                +
                akVs5CaloPositiveSoftMuonByPtBJetTags
              )
            )

akVs5CaloJetBtagging = cms.Sequence(akVs5CaloJetBtaggingIP
            *akVs5CaloJetBtaggingSV
            *akVs5CaloJetBtaggingNegSV
            *akVs5CaloJetBtaggingMu
            )

akVs5CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs5CaloJets"),
        genJetMatch          = cms.InputTag("akVs5Calomatch"),
        genPartonMatch       = cms.InputTag("akVs5Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs5Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs5CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs5CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs5CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs5CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs5CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs5CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs5CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs5CaloJetProbabilityBJetTags"),
            cms.InputTag("akVs5CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akVs5CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs5CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs5CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs5CaloJetID"),
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

akVs5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs5CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak5HiGenJets',
                                                             rParam = 0.5,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("hiSignal"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs5Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs5CaloJetSequence_mc = cms.Sequence(
                                                  akVs5Caloclean
                                                  *
                                                  akVs5Calomatch
                                                  *
                                                  akVs5Caloparton
                                                  *
                                                  akVs5Calocorr
                                                  *
                                                  akVs5CaloJetID
                                                  *
                                                  akVs5CaloPatJetFlavourId
                                                  *
                                                  akVs5CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs5CaloJetBtagging
                                                  *
                                                  akVs5CalopatJetsWithBtagging
                                                  *
                                                  akVs5CaloJetAnalyzer
                                                  )

akVs5CaloJetSequence_data = cms.Sequence(akVs5Calocorr
                                                    *
                                                    akVs5CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs5CaloJetBtagging
                                                    *
                                                    akVs5CalopatJetsWithBtagging
                                                    *
                                                    akVs5CaloJetAnalyzer
                                                    )

akVs5CaloJetSequence_jec = akVs5CaloJetSequence_mc
akVs5CaloJetSequence_mix = akVs5CaloJetSequence_mc

akVs5CaloJetSequence = cms.Sequence(akVs5CaloJetSequence_mix)
