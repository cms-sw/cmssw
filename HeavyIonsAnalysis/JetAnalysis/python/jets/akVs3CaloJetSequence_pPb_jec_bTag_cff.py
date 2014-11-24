

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs3CaloJets"),
    matched = cms.InputTag("ak3HiGenJets")
    )

akVs3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akVs3CaloJets")
                                                        )

akVs3Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs3CaloJets"),
    payload = "AKVs3Calo_HI"
    )

akVs3CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs3CaloJets'))

akVs3Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiGenJets'))

akVs3CalobTagger = bTaggers("akVs3Calo")

#create objects locally since they dont load properly otherwise
akVs3Calomatch = akVs3CalobTagger.match
akVs3Caloparton = akVs3CalobTagger.parton
akVs3CaloPatJetFlavourAssociation = akVs3CalobTagger.PatJetFlavourAssociation
akVs3CaloJetTracksAssociatorAtVertex = akVs3CalobTagger.JetTracksAssociatorAtVertex
akVs3CaloSimpleSecondaryVertexHighEffBJetTags = akVs3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs3CaloSimpleSecondaryVertexHighPurBJetTags = akVs3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs3CaloCombinedSecondaryVertexBJetTags = akVs3CalobTagger.CombinedSecondaryVertexBJetTags
akVs3CaloCombinedSecondaryVertexMVABJetTags = akVs3CalobTagger.CombinedSecondaryVertexMVABJetTags
akVs3CaloJetBProbabilityBJetTags = akVs3CalobTagger.JetBProbabilityBJetTags
akVs3CaloSoftMuonByPtBJetTags = akVs3CalobTagger.SoftMuonByPtBJetTags
akVs3CaloSoftMuonByIP3dBJetTags = akVs3CalobTagger.SoftMuonByIP3dBJetTags
akVs3CaloTrackCountingHighEffBJetTags = akVs3CalobTagger.TrackCountingHighEffBJetTags
akVs3CaloTrackCountingHighPurBJetTags = akVs3CalobTagger.TrackCountingHighPurBJetTags
akVs3CaloPatJetPartonAssociation = akVs3CalobTagger.PatJetPartonAssociation

akVs3CaloImpactParameterTagInfos = akVs3CalobTagger.ImpactParameterTagInfos
akVs3CaloJetProbabilityBJetTags = akVs3CalobTagger.JetProbabilityBJetTags
akVs3CaloPositiveOnlyJetProbabilityJetTags = akVs3CalobTagger.PositiveOnlyJetProbabilityJetTags
akVs3CaloNegativeOnlyJetProbabilityJetTags = akVs3CalobTagger.NegativeOnlyJetProbabilityJetTags
akVs3CaloNegativeTrackCountingHighEffJetTags = akVs3CalobTagger.NegativeTrackCountingHighEffJetTags
akVs3CaloNegativeTrackCountingHighPur = akVs3CalobTagger.NegativeTrackCountingHighPur
akVs3CaloNegativeOnlyJetBProbabilityJetTags = akVs3CalobTagger.NegativeOnlyJetBProbabilityJetTags
akVs3CaloPositiveOnlyJetBProbabilityJetTags = akVs3CalobTagger.PositiveOnlyJetBProbabilityJetTags

akVs3CaloSecondaryVertexTagInfos = akVs3CalobTagger.SecondaryVertexTagInfos
akVs3CaloSimpleSecondaryVertexHighEffBJetTags = akVs3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akVs3CaloSimpleSecondaryVertexHighPurBJetTags = akVs3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akVs3CaloCombinedSecondaryVertexBJetTags = akVs3CalobTagger.CombinedSecondaryVertexBJetTags
akVs3CaloCombinedSecondaryVertexMVABJetTags = akVs3CalobTagger.CombinedSecondaryVertexMVABJetTags

akVs3CaloSecondaryVertexNegativeTagInfos = akVs3CalobTagger.SecondaryVertexNegativeTagInfos
akVs3CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akVs3CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akVs3CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akVs3CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akVs3CaloCombinedSecondaryVertexNegativeBJetTags = akVs3CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akVs3CaloCombinedSecondaryVertexPositiveBJetTags = akVs3CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akVs3CaloSoftMuonTagInfos = akVs3CalobTagger.SoftMuonTagInfos
akVs3CaloSoftMuonBJetTags = akVs3CalobTagger.SoftMuonBJetTags
akVs3CaloSoftMuonByIP3dBJetTags = akVs3CalobTagger.SoftMuonByIP3dBJetTags
akVs3CaloSoftMuonByPtBJetTags = akVs3CalobTagger.SoftMuonByPtBJetTags
akVs3CaloNegativeSoftMuonByPtBJetTags = akVs3CalobTagger.NegativeSoftMuonByPtBJetTags
akVs3CaloPositiveSoftMuonByPtBJetTags = akVs3CalobTagger.PositiveSoftMuonByPtBJetTags

akVs3CaloPatJetFlavourId = cms.Sequence(akVs3CaloPatJetPartonAssociation*akVs3CaloPatJetFlavourAssociation)

akVs3CaloJetBtaggingIP       = cms.Sequence(akVs3CaloImpactParameterTagInfos *
            (akVs3CaloTrackCountingHighEffBJetTags +
             akVs3CaloTrackCountingHighPurBJetTags +
             akVs3CaloJetProbabilityBJetTags +
             akVs3CaloJetBProbabilityBJetTags +
             akVs3CaloPositiveOnlyJetProbabilityJetTags +
             akVs3CaloNegativeOnlyJetProbabilityJetTags +
             akVs3CaloNegativeTrackCountingHighEffJetTags +
             akVs3CaloNegativeTrackCountingHighPur +
             akVs3CaloNegativeOnlyJetBProbabilityJetTags +
             akVs3CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akVs3CaloJetBtaggingSV = cms.Sequence(akVs3CaloImpactParameterTagInfos
            *
            akVs3CaloSecondaryVertexTagInfos
            * (akVs3CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akVs3CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akVs3CaloCombinedSecondaryVertexBJetTags
                +
                akVs3CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akVs3CaloJetBtaggingNegSV = cms.Sequence(akVs3CaloImpactParameterTagInfos
            *
            akVs3CaloSecondaryVertexNegativeTagInfos
            * (akVs3CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akVs3CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akVs3CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akVs3CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akVs3CaloJetBtaggingMu = cms.Sequence(akVs3CaloSoftMuonTagInfos * (akVs3CaloSoftMuonBJetTags
                +
                akVs3CaloSoftMuonByIP3dBJetTags
                +
                akVs3CaloSoftMuonByPtBJetTags
                +
                akVs3CaloNegativeSoftMuonByPtBJetTags
                +
                akVs3CaloPositiveSoftMuonByPtBJetTags
              )
            )

akVs3CaloJetBtagging = cms.Sequence(akVs3CaloJetBtaggingIP
            *akVs3CaloJetBtaggingSV
            *akVs3CaloJetBtaggingNegSV
            *akVs3CaloJetBtaggingMu
            )

akVs3CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs3CaloJets"),
        genJetMatch          = cms.InputTag("akVs3Calomatch"),
        genPartonMatch       = cms.InputTag("akVs3Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs3Calocorr")),
        JetPartonMapSource   = cms.InputTag("akVs3CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs3CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs3CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs3CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs3CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs3CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akVs3CaloJetBProbabilityBJetTags"),
            cms.InputTag("akVs3CaloJetProbabilityBJetTags"),
            cms.InputTag("akVs3CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akVs3CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akVs3CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs3CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs3CaloJetID"),
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

akVs3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs3CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJets',
                                                             rParam = 0.3,
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
                                                             bTagJetName = cms.untracked.string("akVs3Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akVs3CaloJetSequence_mc = cms.Sequence(
                                                  akVs3Caloclean
                                                  *
                                                  akVs3Calomatch
                                                  *
                                                  akVs3Caloparton
                                                  *
                                                  akVs3Calocorr
                                                  *
                                                  akVs3CaloJetID
                                                  *
                                                  akVs3CaloPatJetFlavourId
                                                  *
                                                  akVs3CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akVs3CaloJetBtagging
                                                  *
                                                  akVs3CalopatJetsWithBtagging
                                                  *
                                                  akVs3CaloJetAnalyzer
                                                  )

akVs3CaloJetSequence_data = cms.Sequence(akVs3Calocorr
                                                    *
                                                    akVs3CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akVs3CaloJetBtagging
                                                    *
                                                    akVs3CalopatJetsWithBtagging
                                                    *
                                                    akVs3CaloJetAnalyzer
                                                    )

akVs3CaloJetSequence_jec = akVs3CaloJetSequence_mc
akVs3CaloJetSequence_mix = akVs3CaloJetSequence_mc

akVs3CaloJetSequence = cms.Sequence(akVs3CaloJetSequence_jec)
akVs3CaloJetAnalyzer.genPtMin = cms.untracked.double(1)
