

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu3Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu3CaloJets"),
    matched = cms.InputTag("ak3HiGenJetsCleaned")
    )

akPu3Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu3CaloJets")
                                                        )

akPu3Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu3CaloJets"),
    payload = "AKPu3Calo_HI"
    )

akPu3CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu3CaloJets'))

akPu3Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiGenJetsCleaned'))

akPu3CalobTagger = bTaggers("akPu3Calo")

#create objects locally since they dont load properly otherwise
akPu3Calomatch = akPu3CalobTagger.match
akPu3Caloparton = akPu3CalobTagger.parton
akPu3CaloPatJetFlavourAssociation = akPu3CalobTagger.PatJetFlavourAssociation
akPu3CaloJetTracksAssociatorAtVertex = akPu3CalobTagger.JetTracksAssociatorAtVertex
akPu3CaloSimpleSecondaryVertexHighEffBJetTags = akPu3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu3CaloSimpleSecondaryVertexHighPurBJetTags = akPu3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu3CaloCombinedSecondaryVertexBJetTags = akPu3CalobTagger.CombinedSecondaryVertexBJetTags
akPu3CaloCombinedSecondaryVertexMVABJetTags = akPu3CalobTagger.CombinedSecondaryVertexMVABJetTags
akPu3CaloJetBProbabilityBJetTags = akPu3CalobTagger.JetBProbabilityBJetTags
akPu3CaloSoftMuonByPtBJetTags = akPu3CalobTagger.SoftMuonByPtBJetTags
akPu3CaloSoftMuonByIP3dBJetTags = akPu3CalobTagger.SoftMuonByIP3dBJetTags
akPu3CaloTrackCountingHighEffBJetTags = akPu3CalobTagger.TrackCountingHighEffBJetTags
akPu3CaloTrackCountingHighPurBJetTags = akPu3CalobTagger.TrackCountingHighPurBJetTags
akPu3CaloPatJetPartonAssociation = akPu3CalobTagger.PatJetPartonAssociation

akPu3CaloImpactParameterTagInfos = akPu3CalobTagger.ImpactParameterTagInfos
akPu3CaloJetProbabilityBJetTags = akPu3CalobTagger.JetProbabilityBJetTags
akPu3CaloPositiveOnlyJetProbabilityJetTags = akPu3CalobTagger.PositiveOnlyJetProbabilityJetTags
akPu3CaloNegativeOnlyJetProbabilityJetTags = akPu3CalobTagger.NegativeOnlyJetProbabilityJetTags
akPu3CaloNegativeTrackCountingHighEffJetTags = akPu3CalobTagger.NegativeTrackCountingHighEffJetTags
akPu3CaloNegativeTrackCountingHighPur = akPu3CalobTagger.NegativeTrackCountingHighPur
akPu3CaloNegativeOnlyJetBProbabilityJetTags = akPu3CalobTagger.NegativeOnlyJetBProbabilityJetTags
akPu3CaloPositiveOnlyJetBProbabilityJetTags = akPu3CalobTagger.PositiveOnlyJetBProbabilityJetTags

akPu3CaloSecondaryVertexTagInfos = akPu3CalobTagger.SecondaryVertexTagInfos
akPu3CaloSimpleSecondaryVertexHighEffBJetTags = akPu3CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu3CaloSimpleSecondaryVertexHighPurBJetTags = akPu3CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu3CaloCombinedSecondaryVertexBJetTags = akPu3CalobTagger.CombinedSecondaryVertexBJetTags
akPu3CaloCombinedSecondaryVertexMVABJetTags = akPu3CalobTagger.CombinedSecondaryVertexMVABJetTags

akPu3CaloSecondaryVertexNegativeTagInfos = akPu3CalobTagger.SecondaryVertexNegativeTagInfos
akPu3CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akPu3CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu3CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akPu3CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu3CaloCombinedSecondaryVertexNegativeBJetTags = akPu3CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akPu3CaloCombinedSecondaryVertexPositiveBJetTags = akPu3CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akPu3CaloSoftMuonTagInfos = akPu3CalobTagger.SoftMuonTagInfos
akPu3CaloSoftMuonBJetTags = akPu3CalobTagger.SoftMuonBJetTags
akPu3CaloSoftMuonByIP3dBJetTags = akPu3CalobTagger.SoftMuonByIP3dBJetTags
akPu3CaloSoftMuonByPtBJetTags = akPu3CalobTagger.SoftMuonByPtBJetTags
akPu3CaloNegativeSoftMuonByPtBJetTags = akPu3CalobTagger.NegativeSoftMuonByPtBJetTags
akPu3CaloPositiveSoftMuonByPtBJetTags = akPu3CalobTagger.PositiveSoftMuonByPtBJetTags

akPu3CaloPatJetFlavourId = cms.Sequence(akPu3CaloPatJetPartonAssociation*akPu3CaloPatJetFlavourAssociation)

akPu3CaloJetBtaggingIP       = cms.Sequence(akPu3CaloImpactParameterTagInfos *
            (akPu3CaloTrackCountingHighEffBJetTags +
             akPu3CaloTrackCountingHighPurBJetTags +
             akPu3CaloJetProbabilityBJetTags +
             akPu3CaloJetBProbabilityBJetTags +
             akPu3CaloPositiveOnlyJetProbabilityJetTags +
             akPu3CaloNegativeOnlyJetProbabilityJetTags +
             akPu3CaloNegativeTrackCountingHighEffJetTags +
             akPu3CaloNegativeTrackCountingHighPur +
             akPu3CaloNegativeOnlyJetBProbabilityJetTags +
             akPu3CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu3CaloJetBtaggingSV = cms.Sequence(akPu3CaloImpactParameterTagInfos
            *
            akPu3CaloSecondaryVertexTagInfos
            * (akPu3CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akPu3CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akPu3CaloCombinedSecondaryVertexBJetTags
                +
                akPu3CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akPu3CaloJetBtaggingNegSV = cms.Sequence(akPu3CaloImpactParameterTagInfos
            *
            akPu3CaloSecondaryVertexNegativeTagInfos
            * (akPu3CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu3CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu3CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akPu3CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu3CaloJetBtaggingMu = cms.Sequence(akPu3CaloSoftMuonTagInfos * (akPu3CaloSoftMuonBJetTags
                +
                akPu3CaloSoftMuonByIP3dBJetTags
                +
                akPu3CaloSoftMuonByPtBJetTags
                +
                akPu3CaloNegativeSoftMuonByPtBJetTags
                +
                akPu3CaloPositiveSoftMuonByPtBJetTags
              )
            )

akPu3CaloJetBtagging = cms.Sequence(akPu3CaloJetBtaggingIP
            *akPu3CaloJetBtaggingSV
            *akPu3CaloJetBtaggingNegSV
            *akPu3CaloJetBtaggingMu
            )

akPu3CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu3CaloJets"),
        genJetMatch          = cms.InputTag("akPu3Calomatch"),
        genPartonMatch       = cms.InputTag("akPu3Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu3Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu3CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu3CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu3CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu3CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu3CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu3CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu3CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu3CaloJetProbabilityBJetTags"),
            cms.InputTag("akPu3CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akPu3CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu3CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu3CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu3CaloJetID"),
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

akPu3CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu3CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJetsCleaned',
                                                             rParam = 0.3,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("hiSignal"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akPu3Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu3CaloJetSequence_mc = cms.Sequence(
                                                  akPu3Caloclean
                                                  *
                                                  akPu3Calomatch
                                                  *
                                                  akPu3Caloparton
                                                  *
                                                  akPu3Calocorr
                                                  *
                                                  akPu3CaloJetID
                                                  *
                                                  akPu3CaloPatJetFlavourId
                                                  *
                                                  akPu3CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akPu3CaloJetBtagging
                                                  *
                                                  akPu3CalopatJetsWithBtagging
                                                  *
                                                  akPu3CaloJetAnalyzer
                                                  )

akPu3CaloJetSequence_data = cms.Sequence(akPu3Calocorr
                                                    *
                                                    akPu3CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu3CaloJetBtagging
                                                    *
                                                    akPu3CalopatJetsWithBtagging
                                                    *
                                                    akPu3CaloJetAnalyzer
                                                    )

akPu3CaloJetSequence_jec = akPu3CaloJetSequence_mc
akPu3CaloJetSequence_mix = akPu3CaloJetSequence_mc

akPu3CaloJetSequence = cms.Sequence(akPu3CaloJetSequence_mix)
