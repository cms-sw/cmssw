

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu1Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu1CaloJets"),
    matched = cms.InputTag("ak1HiGenJets")
    )

akPu1Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu1CaloJets")
                                                        )

akPu1Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu1CaloJets"),
    payload = "AKPu1Calo_HI"
    )

akPu1CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu1CaloJets'))

akPu1Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiGenJets'))

akPu1CalobTagger = bTaggers("akPu1Calo")

#create objects locally since they dont load properly otherwise
akPu1Calomatch = akPu1CalobTagger.match
akPu1Caloparton = akPu1CalobTagger.parton
akPu1CaloPatJetFlavourAssociation = akPu1CalobTagger.PatJetFlavourAssociation
akPu1CaloJetTracksAssociatorAtVertex = akPu1CalobTagger.JetTracksAssociatorAtVertex
akPu1CaloSimpleSecondaryVertexHighEffBJetTags = akPu1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu1CaloSimpleSecondaryVertexHighPurBJetTags = akPu1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu1CaloCombinedSecondaryVertexBJetTags = akPu1CalobTagger.CombinedSecondaryVertexBJetTags
akPu1CaloCombinedSecondaryVertexMVABJetTags = akPu1CalobTagger.CombinedSecondaryVertexMVABJetTags
akPu1CaloJetBProbabilityBJetTags = akPu1CalobTagger.JetBProbabilityBJetTags
akPu1CaloSoftMuonByPtBJetTags = akPu1CalobTagger.SoftMuonByPtBJetTags
akPu1CaloSoftMuonByIP3dBJetTags = akPu1CalobTagger.SoftMuonByIP3dBJetTags
akPu1CaloTrackCountingHighEffBJetTags = akPu1CalobTagger.TrackCountingHighEffBJetTags
akPu1CaloTrackCountingHighPurBJetTags = akPu1CalobTagger.TrackCountingHighPurBJetTags
akPu1CaloPatJetPartonAssociation = akPu1CalobTagger.PatJetPartonAssociation

akPu1CaloImpactParameterTagInfos = akPu1CalobTagger.ImpactParameterTagInfos
akPu1CaloJetProbabilityBJetTags = akPu1CalobTagger.JetProbabilityBJetTags
akPu1CaloPositiveOnlyJetProbabilityJetTags = akPu1CalobTagger.PositiveOnlyJetProbabilityJetTags
akPu1CaloNegativeOnlyJetProbabilityJetTags = akPu1CalobTagger.NegativeOnlyJetProbabilityJetTags
akPu1CaloNegativeTrackCountingHighEffJetTags = akPu1CalobTagger.NegativeTrackCountingHighEffJetTags
akPu1CaloNegativeTrackCountingHighPur = akPu1CalobTagger.NegativeTrackCountingHighPur
akPu1CaloNegativeOnlyJetBProbabilityJetTags = akPu1CalobTagger.NegativeOnlyJetBProbabilityJetTags
akPu1CaloPositiveOnlyJetBProbabilityJetTags = akPu1CalobTagger.PositiveOnlyJetBProbabilityJetTags

akPu1CaloSecondaryVertexTagInfos = akPu1CalobTagger.SecondaryVertexTagInfos
akPu1CaloSimpleSecondaryVertexHighEffBJetTags = akPu1CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu1CaloSimpleSecondaryVertexHighPurBJetTags = akPu1CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu1CaloCombinedSecondaryVertexBJetTags = akPu1CalobTagger.CombinedSecondaryVertexBJetTags
akPu1CaloCombinedSecondaryVertexMVABJetTags = akPu1CalobTagger.CombinedSecondaryVertexMVABJetTags

akPu1CaloSecondaryVertexNegativeTagInfos = akPu1CalobTagger.SecondaryVertexNegativeTagInfos
akPu1CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akPu1CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu1CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akPu1CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu1CaloCombinedSecondaryVertexNegativeBJetTags = akPu1CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akPu1CaloCombinedSecondaryVertexPositiveBJetTags = akPu1CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akPu1CaloSoftMuonTagInfos = akPu1CalobTagger.SoftMuonTagInfos
akPu1CaloSoftMuonBJetTags = akPu1CalobTagger.SoftMuonBJetTags
akPu1CaloSoftMuonByIP3dBJetTags = akPu1CalobTagger.SoftMuonByIP3dBJetTags
akPu1CaloSoftMuonByPtBJetTags = akPu1CalobTagger.SoftMuonByPtBJetTags
akPu1CaloNegativeSoftMuonByPtBJetTags = akPu1CalobTagger.NegativeSoftMuonByPtBJetTags
akPu1CaloPositiveSoftMuonByPtBJetTags = akPu1CalobTagger.PositiveSoftMuonByPtBJetTags

akPu1CaloPatJetFlavourId = cms.Sequence(akPu1CaloPatJetPartonAssociation*akPu1CaloPatJetFlavourAssociation)

akPu1CaloJetBtaggingIP       = cms.Sequence(akPu1CaloImpactParameterTagInfos *
            (akPu1CaloTrackCountingHighEffBJetTags +
             akPu1CaloTrackCountingHighPurBJetTags +
             akPu1CaloJetProbabilityBJetTags +
             akPu1CaloJetBProbabilityBJetTags +
             akPu1CaloPositiveOnlyJetProbabilityJetTags +
             akPu1CaloNegativeOnlyJetProbabilityJetTags +
             akPu1CaloNegativeTrackCountingHighEffJetTags +
             akPu1CaloNegativeTrackCountingHighPur +
             akPu1CaloNegativeOnlyJetBProbabilityJetTags +
             akPu1CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu1CaloJetBtaggingSV = cms.Sequence(akPu1CaloImpactParameterTagInfos
            *
            akPu1CaloSecondaryVertexTagInfos
            * (akPu1CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akPu1CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akPu1CaloCombinedSecondaryVertexBJetTags
                +
                akPu1CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akPu1CaloJetBtaggingNegSV = cms.Sequence(akPu1CaloImpactParameterTagInfos
            *
            akPu1CaloSecondaryVertexNegativeTagInfos
            * (akPu1CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu1CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu1CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akPu1CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu1CaloJetBtaggingMu = cms.Sequence(akPu1CaloSoftMuonTagInfos * (akPu1CaloSoftMuonBJetTags
                +
                akPu1CaloSoftMuonByIP3dBJetTags
                +
                akPu1CaloSoftMuonByPtBJetTags
                +
                akPu1CaloNegativeSoftMuonByPtBJetTags
                +
                akPu1CaloPositiveSoftMuonByPtBJetTags
              )
            )

akPu1CaloJetBtagging = cms.Sequence(akPu1CaloJetBtaggingIP
            *akPu1CaloJetBtaggingSV
            *akPu1CaloJetBtaggingNegSV
            *akPu1CaloJetBtaggingMu
            )

akPu1CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu1CaloJets"),
        genJetMatch          = cms.InputTag("akPu1Calomatch"),
        genPartonMatch       = cms.InputTag("akPu1Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu1Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu1CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu1CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu1CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu1CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu1CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu1CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu1CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu1CaloJetProbabilityBJetTags"),
            cms.InputTag("akPu1CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akPu1CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu1CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu1CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu1CaloJetID"),
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

akPu1CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu1CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akPu1Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu1CaloJetSequence_mc = cms.Sequence(
                                                  akPu1Caloclean
                                                  *
                                                  akPu1Calomatch
                                                  *
                                                  akPu1Caloparton
                                                  *
                                                  akPu1Calocorr
                                                  *
                                                  akPu1CaloJetID
                                                  *
                                                  akPu1CaloPatJetFlavourId
                                                  *
                                                  akPu1CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akPu1CaloJetBtagging
                                                  *
                                                  akPu1CalopatJetsWithBtagging
                                                  *
                                                  akPu1CaloJetAnalyzer
                                                  )

akPu1CaloJetSequence_data = cms.Sequence(akPu1Calocorr
                                                    *
                                                    akPu1CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu1CaloJetBtagging
                                                    *
                                                    akPu1CalopatJetsWithBtagging
                                                    *
                                                    akPu1CaloJetAnalyzer
                                                    )

akPu1CaloJetSequence_jec = akPu1CaloJetSequence_mc
akPu1CaloJetSequence_mix = akPu1CaloJetSequence_mc

akPu1CaloJetSequence = cms.Sequence(akPu1CaloJetSequence_data)
