

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu7Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu7CaloJets"),
    matched = cms.InputTag("ak7HiGenJets")
    )

akPu7Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu7CaloJets")
                                                        )

akPu7Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu7CaloJets"),
    payload = "AKPu7Calo_HI"
    )

akPu7CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu7CaloJets'))

akPu7Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

akPu7CalobTagger = bTaggers("akPu7Calo")

#create objects locally since they dont load properly otherwise
akPu7Calomatch = akPu7CalobTagger.match
akPu7Caloparton = akPu7CalobTagger.parton
akPu7CaloPatJetFlavourAssociation = akPu7CalobTagger.PatJetFlavourAssociation
akPu7CaloJetTracksAssociatorAtVertex = akPu7CalobTagger.JetTracksAssociatorAtVertex
akPu7CaloSimpleSecondaryVertexHighEffBJetTags = akPu7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu7CaloSimpleSecondaryVertexHighPurBJetTags = akPu7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu7CaloCombinedSecondaryVertexBJetTags = akPu7CalobTagger.CombinedSecondaryVertexBJetTags
akPu7CaloCombinedSecondaryVertexMVABJetTags = akPu7CalobTagger.CombinedSecondaryVertexMVABJetTags
akPu7CaloJetBProbabilityBJetTags = akPu7CalobTagger.JetBProbabilityBJetTags
akPu7CaloSoftMuonByPtBJetTags = akPu7CalobTagger.SoftMuonByPtBJetTags
akPu7CaloSoftMuonByIP3dBJetTags = akPu7CalobTagger.SoftMuonByIP3dBJetTags
akPu7CaloTrackCountingHighEffBJetTags = akPu7CalobTagger.TrackCountingHighEffBJetTags
akPu7CaloTrackCountingHighPurBJetTags = akPu7CalobTagger.TrackCountingHighPurBJetTags
akPu7CaloPatJetPartonAssociation = akPu7CalobTagger.PatJetPartonAssociation

akPu7CaloImpactParameterTagInfos = akPu7CalobTagger.ImpactParameterTagInfos
akPu7CaloJetProbabilityBJetTags = akPu7CalobTagger.JetProbabilityBJetTags
akPu7CaloPositiveOnlyJetProbabilityJetTags = akPu7CalobTagger.PositiveOnlyJetProbabilityJetTags
akPu7CaloNegativeOnlyJetProbabilityJetTags = akPu7CalobTagger.NegativeOnlyJetProbabilityJetTags
akPu7CaloNegativeTrackCountingHighEffJetTags = akPu7CalobTagger.NegativeTrackCountingHighEffJetTags
akPu7CaloNegativeTrackCountingHighPur = akPu7CalobTagger.NegativeTrackCountingHighPur
akPu7CaloNegativeOnlyJetBProbabilityJetTags = akPu7CalobTagger.NegativeOnlyJetBProbabilityJetTags
akPu7CaloPositiveOnlyJetBProbabilityJetTags = akPu7CalobTagger.PositiveOnlyJetBProbabilityJetTags

akPu7CaloSecondaryVertexTagInfos = akPu7CalobTagger.SecondaryVertexTagInfos
akPu7CaloSimpleSecondaryVertexHighEffBJetTags = akPu7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu7CaloSimpleSecondaryVertexHighPurBJetTags = akPu7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu7CaloCombinedSecondaryVertexBJetTags = akPu7CalobTagger.CombinedSecondaryVertexBJetTags
akPu7CaloCombinedSecondaryVertexMVABJetTags = akPu7CalobTagger.CombinedSecondaryVertexMVABJetTags

akPu7CaloSecondaryVertexNegativeTagInfos = akPu7CalobTagger.SecondaryVertexNegativeTagInfos
akPu7CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akPu7CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu7CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akPu7CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu7CaloCombinedSecondaryVertexNegativeBJetTags = akPu7CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akPu7CaloCombinedSecondaryVertexPositiveBJetTags = akPu7CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akPu7CaloSoftMuonTagInfos = akPu7CalobTagger.SoftMuonTagInfos
akPu7CaloSoftMuonBJetTags = akPu7CalobTagger.SoftMuonBJetTags
akPu7CaloSoftMuonByIP3dBJetTags = akPu7CalobTagger.SoftMuonByIP3dBJetTags
akPu7CaloSoftMuonByPtBJetTags = akPu7CalobTagger.SoftMuonByPtBJetTags
akPu7CaloNegativeSoftMuonByPtBJetTags = akPu7CalobTagger.NegativeSoftMuonByPtBJetTags
akPu7CaloPositiveSoftMuonByPtBJetTags = akPu7CalobTagger.PositiveSoftMuonByPtBJetTags

akPu7CaloPatJetFlavourId = cms.Sequence(akPu7CaloPatJetPartonAssociation*akPu7CaloPatJetFlavourAssociation)

akPu7CaloJetBtaggingIP       = cms.Sequence(akPu7CaloImpactParameterTagInfos *
            (akPu7CaloTrackCountingHighEffBJetTags +
             akPu7CaloTrackCountingHighPurBJetTags +
             akPu7CaloJetProbabilityBJetTags +
             akPu7CaloJetBProbabilityBJetTags +
             akPu7CaloPositiveOnlyJetProbabilityJetTags +
             akPu7CaloNegativeOnlyJetProbabilityJetTags +
             akPu7CaloNegativeTrackCountingHighEffJetTags +
             akPu7CaloNegativeTrackCountingHighPur +
             akPu7CaloNegativeOnlyJetBProbabilityJetTags +
             akPu7CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu7CaloJetBtaggingSV = cms.Sequence(akPu7CaloImpactParameterTagInfos
            *
            akPu7CaloSecondaryVertexTagInfos
            * (akPu7CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akPu7CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akPu7CaloCombinedSecondaryVertexBJetTags
                +
                akPu7CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akPu7CaloJetBtaggingNegSV = cms.Sequence(akPu7CaloImpactParameterTagInfos
            *
            akPu7CaloSecondaryVertexNegativeTagInfos
            * (akPu7CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu7CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu7CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akPu7CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu7CaloJetBtaggingMu = cms.Sequence(akPu7CaloSoftMuonTagInfos * (akPu7CaloSoftMuonBJetTags
                +
                akPu7CaloSoftMuonByIP3dBJetTags
                +
                akPu7CaloSoftMuonByPtBJetTags
                +
                akPu7CaloNegativeSoftMuonByPtBJetTags
                +
                akPu7CaloPositiveSoftMuonByPtBJetTags
              )
            )

akPu7CaloJetBtagging = cms.Sequence(akPu7CaloJetBtaggingIP
            *akPu7CaloJetBtaggingSV
            *akPu7CaloJetBtaggingNegSV
            *akPu7CaloJetBtaggingMu
            )

akPu7CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu7CaloJets"),
        genJetMatch          = cms.InputTag("akPu7Calomatch"),
        genPartonMatch       = cms.InputTag("akPu7Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu7Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu7CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu7CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu7CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu7CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu7CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu7CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu7CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu7CaloJetProbabilityBJetTags"),
            cms.InputTag("akPu7CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akPu7CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu7CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu7CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu7CaloJetID"),
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

akPu7CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu7CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akPu7Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu7CaloJetSequence_mc = cms.Sequence(
                                                  akPu7Caloclean
                                                  *
                                                  akPu7Calomatch
                                                  *
                                                  akPu7Caloparton
                                                  *
                                                  akPu7Calocorr
                                                  *
                                                  akPu7CaloJetID
                                                  *
                                                  akPu7CaloPatJetFlavourId
                                                  *
                                                  akPu7CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akPu7CaloJetBtagging
                                                  *
                                                  akPu7CalopatJetsWithBtagging
                                                  *
                                                  akPu7CaloJetAnalyzer
                                                  )

akPu7CaloJetSequence_data = cms.Sequence(akPu7Calocorr
                                                    *
                                                    akPu7CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu7CaloJetBtagging
                                                    *
                                                    akPu7CalopatJetsWithBtagging
                                                    *
                                                    akPu7CaloJetAnalyzer
                                                    )

akPu7CaloJetSequence_jec = akPu7CaloJetSequence_mc
akPu7CaloJetSequence_mix = akPu7CaloJetSequence_mc

akPu7CaloJetSequence = cms.Sequence(akPu7CaloJetSequence_data)
