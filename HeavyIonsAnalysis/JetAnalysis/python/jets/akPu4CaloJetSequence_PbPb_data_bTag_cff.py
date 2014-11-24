

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu4Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu4CaloJets"),
    matched = cms.InputTag("ak4HiGenJetsCleaned")
    )

akPu4Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu4CaloJets")
                                                        )

akPu4Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu4CaloJets"),
    payload = "AKPu4Calo_HI"
    )

akPu4CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu4CaloJets'))

akPu4Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJetsCleaned'))

akPu4CalobTagger = bTaggers("akPu4Calo")

#create objects locally since they dont load properly otherwise
akPu4Calomatch = akPu4CalobTagger.match
akPu4Caloparton = akPu4CalobTagger.parton
akPu4CaloPatJetFlavourAssociation = akPu4CalobTagger.PatJetFlavourAssociation
akPu4CaloJetTracksAssociatorAtVertex = akPu4CalobTagger.JetTracksAssociatorAtVertex
akPu4CaloSimpleSecondaryVertexHighEffBJetTags = akPu4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu4CaloSimpleSecondaryVertexHighPurBJetTags = akPu4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu4CaloCombinedSecondaryVertexBJetTags = akPu4CalobTagger.CombinedSecondaryVertexBJetTags
akPu4CaloCombinedSecondaryVertexMVABJetTags = akPu4CalobTagger.CombinedSecondaryVertexMVABJetTags
akPu4CaloJetBProbabilityBJetTags = akPu4CalobTagger.JetBProbabilityBJetTags
akPu4CaloSoftMuonByPtBJetTags = akPu4CalobTagger.SoftMuonByPtBJetTags
akPu4CaloSoftMuonByIP3dBJetTags = akPu4CalobTagger.SoftMuonByIP3dBJetTags
akPu4CaloTrackCountingHighEffBJetTags = akPu4CalobTagger.TrackCountingHighEffBJetTags
akPu4CaloTrackCountingHighPurBJetTags = akPu4CalobTagger.TrackCountingHighPurBJetTags
akPu4CaloPatJetPartonAssociation = akPu4CalobTagger.PatJetPartonAssociation

akPu4CaloImpactParameterTagInfos = akPu4CalobTagger.ImpactParameterTagInfos
akPu4CaloJetProbabilityBJetTags = akPu4CalobTagger.JetProbabilityBJetTags
akPu4CaloPositiveOnlyJetProbabilityJetTags = akPu4CalobTagger.PositiveOnlyJetProbabilityJetTags
akPu4CaloNegativeOnlyJetProbabilityJetTags = akPu4CalobTagger.NegativeOnlyJetProbabilityJetTags
akPu4CaloNegativeTrackCountingHighEffJetTags = akPu4CalobTagger.NegativeTrackCountingHighEffJetTags
akPu4CaloNegativeTrackCountingHighPur = akPu4CalobTagger.NegativeTrackCountingHighPur
akPu4CaloNegativeOnlyJetBProbabilityJetTags = akPu4CalobTagger.NegativeOnlyJetBProbabilityJetTags
akPu4CaloPositiveOnlyJetBProbabilityJetTags = akPu4CalobTagger.PositiveOnlyJetBProbabilityJetTags

akPu4CaloSecondaryVertexTagInfos = akPu4CalobTagger.SecondaryVertexTagInfos
akPu4CaloSimpleSecondaryVertexHighEffBJetTags = akPu4CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu4CaloSimpleSecondaryVertexHighPurBJetTags = akPu4CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu4CaloCombinedSecondaryVertexBJetTags = akPu4CalobTagger.CombinedSecondaryVertexBJetTags
akPu4CaloCombinedSecondaryVertexMVABJetTags = akPu4CalobTagger.CombinedSecondaryVertexMVABJetTags

akPu4CaloSecondaryVertexNegativeTagInfos = akPu4CalobTagger.SecondaryVertexNegativeTagInfos
akPu4CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akPu4CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu4CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akPu4CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu4CaloCombinedSecondaryVertexNegativeBJetTags = akPu4CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akPu4CaloCombinedSecondaryVertexPositiveBJetTags = akPu4CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akPu4CaloSoftMuonTagInfos = akPu4CalobTagger.SoftMuonTagInfos
akPu4CaloSoftMuonBJetTags = akPu4CalobTagger.SoftMuonBJetTags
akPu4CaloSoftMuonByIP3dBJetTags = akPu4CalobTagger.SoftMuonByIP3dBJetTags
akPu4CaloSoftMuonByPtBJetTags = akPu4CalobTagger.SoftMuonByPtBJetTags
akPu4CaloNegativeSoftMuonByPtBJetTags = akPu4CalobTagger.NegativeSoftMuonByPtBJetTags
akPu4CaloPositiveSoftMuonByPtBJetTags = akPu4CalobTagger.PositiveSoftMuonByPtBJetTags

akPu4CaloPatJetFlavourId = cms.Sequence(akPu4CaloPatJetPartonAssociation*akPu4CaloPatJetFlavourAssociation)

akPu4CaloJetBtaggingIP       = cms.Sequence(akPu4CaloImpactParameterTagInfos *
            (akPu4CaloTrackCountingHighEffBJetTags +
             akPu4CaloTrackCountingHighPurBJetTags +
             akPu4CaloJetProbabilityBJetTags +
             akPu4CaloJetBProbabilityBJetTags +
             akPu4CaloPositiveOnlyJetProbabilityJetTags +
             akPu4CaloNegativeOnlyJetProbabilityJetTags +
             akPu4CaloNegativeTrackCountingHighEffJetTags +
             akPu4CaloNegativeTrackCountingHighPur +
             akPu4CaloNegativeOnlyJetBProbabilityJetTags +
             akPu4CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu4CaloJetBtaggingSV = cms.Sequence(akPu4CaloImpactParameterTagInfos
            *
            akPu4CaloSecondaryVertexTagInfos
            * (akPu4CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akPu4CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akPu4CaloCombinedSecondaryVertexBJetTags
                +
                akPu4CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akPu4CaloJetBtaggingNegSV = cms.Sequence(akPu4CaloImpactParameterTagInfos
            *
            akPu4CaloSecondaryVertexNegativeTagInfos
            * (akPu4CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu4CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu4CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akPu4CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu4CaloJetBtaggingMu = cms.Sequence(akPu4CaloSoftMuonTagInfos * (akPu4CaloSoftMuonBJetTags
                +
                akPu4CaloSoftMuonByIP3dBJetTags
                +
                akPu4CaloSoftMuonByPtBJetTags
                +
                akPu4CaloNegativeSoftMuonByPtBJetTags
                +
                akPu4CaloPositiveSoftMuonByPtBJetTags
              )
            )

akPu4CaloJetBtagging = cms.Sequence(akPu4CaloJetBtaggingIP
            *akPu4CaloJetBtaggingSV
            *akPu4CaloJetBtaggingNegSV
            *akPu4CaloJetBtaggingMu
            )

akPu4CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu4CaloJets"),
        genJetMatch          = cms.InputTag("akPu4Calomatch"),
        genPartonMatch       = cms.InputTag("akPu4Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu4Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu4CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu4CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu4CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu4CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu4CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu4CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu4CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu4CaloJetProbabilityBJetTags"),
            cms.InputTag("akPu4CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akPu4CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu4CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu4CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu4CaloJetID"),
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

akPu4CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu4CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak4HiGenJetsCleaned',
                                                             rParam = 0.4,
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
                                                             bTagJetName = cms.untracked.string("akPu4Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu4CaloJetSequence_mc = cms.Sequence(
                                                  akPu4Caloclean
                                                  *
                                                  akPu4Calomatch
                                                  *
                                                  akPu4Caloparton
                                                  *
                                                  akPu4Calocorr
                                                  *
                                                  akPu4CaloJetID
                                                  *
                                                  akPu4CaloPatJetFlavourId
                                                  *
                                                  akPu4CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akPu4CaloJetBtagging
                                                  *
                                                  akPu4CalopatJetsWithBtagging
                                                  *
                                                  akPu4CaloJetAnalyzer
                                                  )

akPu4CaloJetSequence_data = cms.Sequence(akPu4Calocorr
                                                    *
                                                    akPu4CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu4CaloJetBtagging
                                                    *
                                                    akPu4CalopatJetsWithBtagging
                                                    *
                                                    akPu4CaloJetAnalyzer
                                                    )

akPu4CaloJetSequence_jec = akPu4CaloJetSequence_mc
akPu4CaloJetSequence_mix = akPu4CaloJetSequence_mc

akPu4CaloJetSequence = cms.Sequence(akPu4CaloJetSequence_data)
