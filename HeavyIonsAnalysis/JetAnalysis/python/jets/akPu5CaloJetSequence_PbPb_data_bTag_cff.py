

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu5Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu5CaloJets"),
    matched = cms.InputTag("ak5HiGenJetsCleaned")
    )

akPu5Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu5CaloJets")
                                                        )

akPu5Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu5CaloJets"),
    payload = "AKPu5Calo_HI"
    )

akPu5CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu5CaloJets'))

akPu5Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiGenJetsCleaned'))

akPu5CalobTagger = bTaggers("akPu5Calo")

#create objects locally since they dont load properly otherwise
akPu5Calomatch = akPu5CalobTagger.match
akPu5Caloparton = akPu5CalobTagger.parton
akPu5CaloPatJetFlavourAssociation = akPu5CalobTagger.PatJetFlavourAssociation
akPu5CaloJetTracksAssociatorAtVertex = akPu5CalobTagger.JetTracksAssociatorAtVertex
akPu5CaloSimpleSecondaryVertexHighEffBJetTags = akPu5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu5CaloSimpleSecondaryVertexHighPurBJetTags = akPu5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu5CaloCombinedSecondaryVertexBJetTags = akPu5CalobTagger.CombinedSecondaryVertexBJetTags
akPu5CaloCombinedSecondaryVertexMVABJetTags = akPu5CalobTagger.CombinedSecondaryVertexMVABJetTags
akPu5CaloJetBProbabilityBJetTags = akPu5CalobTagger.JetBProbabilityBJetTags
akPu5CaloSoftMuonByPtBJetTags = akPu5CalobTagger.SoftMuonByPtBJetTags
akPu5CaloSoftMuonByIP3dBJetTags = akPu5CalobTagger.SoftMuonByIP3dBJetTags
akPu5CaloTrackCountingHighEffBJetTags = akPu5CalobTagger.TrackCountingHighEffBJetTags
akPu5CaloTrackCountingHighPurBJetTags = akPu5CalobTagger.TrackCountingHighPurBJetTags
akPu5CaloPatJetPartonAssociation = akPu5CalobTagger.PatJetPartonAssociation

akPu5CaloImpactParameterTagInfos = akPu5CalobTagger.ImpactParameterTagInfos
akPu5CaloJetProbabilityBJetTags = akPu5CalobTagger.JetProbabilityBJetTags
akPu5CaloPositiveOnlyJetProbabilityJetTags = akPu5CalobTagger.PositiveOnlyJetProbabilityJetTags
akPu5CaloNegativeOnlyJetProbabilityJetTags = akPu5CalobTagger.NegativeOnlyJetProbabilityJetTags
akPu5CaloNegativeTrackCountingHighEffJetTags = akPu5CalobTagger.NegativeTrackCountingHighEffJetTags
akPu5CaloNegativeTrackCountingHighPur = akPu5CalobTagger.NegativeTrackCountingHighPur
akPu5CaloNegativeOnlyJetBProbabilityJetTags = akPu5CalobTagger.NegativeOnlyJetBProbabilityJetTags
akPu5CaloPositiveOnlyJetBProbabilityJetTags = akPu5CalobTagger.PositiveOnlyJetBProbabilityJetTags

akPu5CaloSecondaryVertexTagInfos = akPu5CalobTagger.SecondaryVertexTagInfos
akPu5CaloSimpleSecondaryVertexHighEffBJetTags = akPu5CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu5CaloSimpleSecondaryVertexHighPurBJetTags = akPu5CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu5CaloCombinedSecondaryVertexBJetTags = akPu5CalobTagger.CombinedSecondaryVertexBJetTags
akPu5CaloCombinedSecondaryVertexMVABJetTags = akPu5CalobTagger.CombinedSecondaryVertexMVABJetTags

akPu5CaloSecondaryVertexNegativeTagInfos = akPu5CalobTagger.SecondaryVertexNegativeTagInfos
akPu5CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akPu5CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu5CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akPu5CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu5CaloCombinedSecondaryVertexNegativeBJetTags = akPu5CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akPu5CaloCombinedSecondaryVertexPositiveBJetTags = akPu5CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akPu5CaloSoftMuonTagInfos = akPu5CalobTagger.SoftMuonTagInfos
akPu5CaloSoftMuonBJetTags = akPu5CalobTagger.SoftMuonBJetTags
akPu5CaloSoftMuonByIP3dBJetTags = akPu5CalobTagger.SoftMuonByIP3dBJetTags
akPu5CaloSoftMuonByPtBJetTags = akPu5CalobTagger.SoftMuonByPtBJetTags
akPu5CaloNegativeSoftMuonByPtBJetTags = akPu5CalobTagger.NegativeSoftMuonByPtBJetTags
akPu5CaloPositiveSoftMuonByPtBJetTags = akPu5CalobTagger.PositiveSoftMuonByPtBJetTags

akPu5CaloPatJetFlavourId = cms.Sequence(akPu5CaloPatJetPartonAssociation*akPu5CaloPatJetFlavourAssociation)

akPu5CaloJetBtaggingIP       = cms.Sequence(akPu5CaloImpactParameterTagInfos *
            (akPu5CaloTrackCountingHighEffBJetTags +
             akPu5CaloTrackCountingHighPurBJetTags +
             akPu5CaloJetProbabilityBJetTags +
             akPu5CaloJetBProbabilityBJetTags +
             akPu5CaloPositiveOnlyJetProbabilityJetTags +
             akPu5CaloNegativeOnlyJetProbabilityJetTags +
             akPu5CaloNegativeTrackCountingHighEffJetTags +
             akPu5CaloNegativeTrackCountingHighPur +
             akPu5CaloNegativeOnlyJetBProbabilityJetTags +
             akPu5CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu5CaloJetBtaggingSV = cms.Sequence(akPu5CaloImpactParameterTagInfos
            *
            akPu5CaloSecondaryVertexTagInfos
            * (akPu5CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akPu5CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akPu5CaloCombinedSecondaryVertexBJetTags
                +
                akPu5CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akPu5CaloJetBtaggingNegSV = cms.Sequence(akPu5CaloImpactParameterTagInfos
            *
            akPu5CaloSecondaryVertexNegativeTagInfos
            * (akPu5CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu5CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu5CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akPu5CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu5CaloJetBtaggingMu = cms.Sequence(akPu5CaloSoftMuonTagInfos * (akPu5CaloSoftMuonBJetTags
                +
                akPu5CaloSoftMuonByIP3dBJetTags
                +
                akPu5CaloSoftMuonByPtBJetTags
                +
                akPu5CaloNegativeSoftMuonByPtBJetTags
                +
                akPu5CaloPositiveSoftMuonByPtBJetTags
              )
            )

akPu5CaloJetBtagging = cms.Sequence(akPu5CaloJetBtaggingIP
            *akPu5CaloJetBtaggingSV
            *akPu5CaloJetBtaggingNegSV
            *akPu5CaloJetBtaggingMu
            )

akPu5CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu5CaloJets"),
        genJetMatch          = cms.InputTag("akPu5Calomatch"),
        genPartonMatch       = cms.InputTag("akPu5Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu5Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu5CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu5CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu5CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu5CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu5CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu5CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu5CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu5CaloJetProbabilityBJetTags"),
            cms.InputTag("akPu5CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akPu5CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu5CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu5CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu5CaloJetID"),
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

akPu5CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu5CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak5HiGenJetsCleaned',
                                                             rParam = 0.5,
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
                                                             bTagJetName = cms.untracked.string("akPu5Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu5CaloJetSequence_mc = cms.Sequence(
                                                  akPu5Caloclean
                                                  *
                                                  akPu5Calomatch
                                                  *
                                                  akPu5Caloparton
                                                  *
                                                  akPu5Calocorr
                                                  *
                                                  akPu5CaloJetID
                                                  *
                                                  akPu5CaloPatJetFlavourId
                                                  *
                                                  akPu5CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akPu5CaloJetBtagging
                                                  *
                                                  akPu5CalopatJetsWithBtagging
                                                  *
                                                  akPu5CaloJetAnalyzer
                                                  )

akPu5CaloJetSequence_data = cms.Sequence(akPu5Calocorr
                                                    *
                                                    akPu5CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu5CaloJetBtagging
                                                    *
                                                    akPu5CalopatJetsWithBtagging
                                                    *
                                                    akPu5CaloJetAnalyzer
                                                    )

akPu5CaloJetSequence_jec = akPu5CaloJetSequence_mc
akPu5CaloJetSequence_mix = akPu5CaloJetSequence_mc

akPu5CaloJetSequence = cms.Sequence(akPu5CaloJetSequence_data)
