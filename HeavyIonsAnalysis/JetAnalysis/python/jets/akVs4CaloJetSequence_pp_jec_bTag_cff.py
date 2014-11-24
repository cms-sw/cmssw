

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs4CaloJets"),
    matched = cms.InputTag("ak4HiGenJets")
    )

akVs4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs4CaloJets")
                                                        )

akVs4Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs4CaloJets"),
    payload = "AKVs4Calo_HI"
    )

akVs4CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs4CaloJets'))

akVs4Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJets'))

akVs4CalobTagger = bTaggers("akVs4Calo")

#create objects locally since they dont load properly otherwise
akVs4Calomatch = akVs4CalobTagger.match
akVs4Caloparton = akVs4CalobTagger.parton
akVs4CaloPatJetFlavourAssociation = akVs4CalobTagger.PatJetFlavourAssociation
akVs4CaloJetTracksAssociatorAtVertex = akVs4CalobTagger.JetTracksAssociatorAtVertex
akVs4CaloSimpleSecondaryVertexHighEffBJetTags = akVs4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs4CaloSimpleSecondaryVertexHighPurBJetTags = akVs4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs4CaloCombinedSecondaryVertexBJetTags = akVs4CalobTagger.CombinedSecondaryVertexBJetTags
akVs4CaloCombinedSecondaryVertexMVABJetTags = akVs4CalobTagger.CombinedSecondaryVertexMVABJetTags
akVs4CaloJetBProbabilityBJetTags = akVs4CalobTagger.JetBProbabilityBJetTags
akVs4CaloSoftMuonByPtBJetTags = akVs4CalobTagger.SoftMuonByPtBJetTags
akVs4CaloSoftMuonByIP3dBJetTags = akVs4CalobTagger.SoftMuonByIP3dBJetTags
akVs4CaloTrackCountingHighEffBJetTags = akVs4CalobTagger.TrackCountingHighEffBJetTags
akVs4CaloTrackCountingHighPurBJetTags = akVs4CalobTagger.TrackCountingHighPurBJetTags
akVs4CaloPatJetPartonAssociation = akVs4CalobTagger.PatJetPartonAssociation

akVs4CaloImpactParameterTagInfos = akVs4CalobTagger.ImpactParameterTagInfos
akVs4CaloJetProbabilityBJetTags = akVs4CalobTagger.JetProbabilityBJetTags
akVs4CaloPositiveOnlyJetProbabilityJetTags = akVs4CalobTagger.PositiveOnlyJetProbabilityJetTags
akVs4CaloNegativeOnlyJetProbabilityJetTags = akVs4CalobTagger.NegativeOnlyJetProbabilityJetTags
akVs4CaloNegativeTrackCountingHighEffJetTags = akVs4CalobTagger.NegativeTrackCountingHighEffJetTags
akVs4CaloNegativeTrackCountingHighPur = akVs4CalobTagger.NegativeTrackCountingHighPur
akVs4CaloNegativeOnlyJetBProbabilityJetTags = akVs4CalobTagger.NegativeOnlyJetBProbabilityJetTags
akVs4CaloPositiveOnlyJetBProbabilityJetTags = akVs4CalobTagger.PositiveOnlyJetBProbabilityJetTags

akVs4CaloSecondaryVertexTagInfos = akVs4CalobTagger.SecondaryVertexTagInfos
akVs4CaloSimpleSecondaryVertexHighEffBJetTags = akVs4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs4CaloSimpleSecondaryVertexHighPurBJetTags = akVs4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs4CaloCombinedSecondaryVertexBJetTags = akVs4CalobTagger.CombinedSecondaryVertexBJetTags
akVs4CaloCombinedSecondaryVertexMVABJetTags = akVs4CalobTagger.CombinedSecondaryVertexMVABJetTags

akVs4CaloSecondaryVertexNegativeTagInfos = akVs4CalobTagger.SecondaryVertexNegativeTagInfos
akVs4CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akVs4CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs4CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akVs4CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs4CaloCombinedSecondaryVertexNegativeBJetTags = akVs4CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akVs4CaloCombinedSecondaryVertexPositiveBJetTags = akVs4CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akVs4CaloSoftMuonTagInfos = akVs4CalobTagger.SoftMuonTagInfos
akVs4CaloSoftMuonBJetTags = akVs4CalobTagger.SoftMuonBJetTags
akVs4CaloSoftMuonByIP3dBJetTags = akVs4CalobTagger.SoftMuonByIP3dBJetTags
akVs4CaloSoftMuonByPtBJetTags = akVs4CalobTagger.SoftMuonByPtBJetTags
akVs4CaloNegativeSoftMuonByPtBJetTags = akVs4CalobTagger.NegativeSoftMuonByPtBJetTags
akVs4CaloPositiveSoftMuonByPtBJetTags = akVs4CalobTagger.PositiveSoftMuonByPtBJetTags

akVs4CaloPatJetFlavourId = cms.Sequence(akVs4CaloPatJetPartonAssociation*akVs4CaloPatJetFlavourAssociation)

akVs4CaloJetBtaggingIP       = cms.Sequence(akVs4CaloImpactParameterTagInfos *
            (akVs4CaloTrackCountingHighEffBJetTags +
             akVs4CaloTrackCountingHighPurBJetTags +
             akVs4CaloJetProbabilityBJetTags +
             akVs4CaloJetBProbabilityBJetTags +
             akVs4CaloPositiveOnlyJetProbabilityJetTags +
             akVs4CaloNegativeOnlyJetProbabilityJetTags +
             akVs4CaloNegativeTrackCountingHighEffJetTags +
             akVs4CaloNegativeTrackCountingHighPur +
             akVs4CaloNegativeOnlyJetBProbabilityJetTags +
             akVs4CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs4CaloJetBtaggingSV = cms.Sequence(akVs4CaloImpactParameterTagInfos
            *
            akVs4CaloSecondaryVertexTagInfos
            * (akVs4CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs4CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs4CaloCombinedSecondaryVertexBJetTags
                +
                akVs4CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akVs4CaloJetBtaggingNegSV = cms.Sequence(akVs4CaloImpactParameterTagInfos
            *
            akVs4CaloSecondaryVertexNegativeTagInfos
            * (akVs4CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs4CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs4CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akVs4CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs4CaloJetBtaggingMu = cms.Sequence(akVs4CaloSoftMuonTagInfos * (akVs4CaloSoftMuonBJetTags
                +
                akVs4CaloSoftMuonByIP3dBJetTags
                +
                akVs4CaloSoftMuonByPtBJetTags
                +
                akVs4CaloNegativeSoftMuonByPtBJetTags
                +
                akVs4CaloPositiveSoftMuonByPtBJetTags
              )
            )

akVs4CaloJetBtagging = cms.Sequence(akVs4CaloJetBtaggingIP
            *akVs4CaloJetBtaggingSV
            *akVs4CaloJetBtaggingNegSV
            *akVs4CaloJetBtaggingMu
            )

akVs4CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs4CaloJets"),
        genJetMatch          = cms.InputTag("akVs4Calomatch"),
        genPartonMatch       = cms.InputTag("akVs4Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs4Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs4CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs4CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs4CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs4CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs4CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs4CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs4CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs4CaloJetProbabilityBJetTags"),
            cms.InputTag("akVs4CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akVs4CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs4CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs4CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs4CaloJetID"),
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

akVs4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs4CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akVs4Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs4CaloJetSequence_mc = cms.Sequence(
                                                  akVs4Caloclean
                                                  *
                                                  akVs4Calomatch
                                                  *
                                                  akVs4Caloparton
                                                  *
                                                  akVs4Calocorr
                                                  *
                                                  akVs4CaloJetID
                                                  *
                                                  akVs4CaloPatJetFlavourId
                                                  *
                                                  akVs4CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs4CaloJetBtagging
                                                  *
                                                  akVs4CalopatJetsWithBtagging
                                                  *
                                                  akVs4CaloJetAnalyzer
                                                  )

akVs4CaloJetSequence_data = cms.Sequence(akVs4Calocorr
                                                    *
                                                    akVs4CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs4CaloJetBtagging
                                                    *
                                                    akVs4CalopatJetsWithBtagging
                                                    *
                                                    akVs4CaloJetAnalyzer
                                                    )

akVs4CaloJetSequence_jec = akVs4CaloJetSequence_mc
akVs4CaloJetSequence_mix = akVs4CaloJetSequence_mc

akVs4CaloJetSequence = cms.Sequence(akVs4CaloJetSequence_jec)
akVs4CaloJetAnalyzer.genPtMin = cms.untracked.double(1)
