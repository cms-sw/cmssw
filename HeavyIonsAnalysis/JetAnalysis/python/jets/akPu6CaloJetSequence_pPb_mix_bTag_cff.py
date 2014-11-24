

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu6CaloJets"),
    matched = cms.InputTag("ak6HiGenJets")
    )

akPu6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu6CaloJets")
                                                        )

akPu6Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu6CaloJets"),
    payload = "AKPu6Calo_HI"
    )

akPu6CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu6CaloJets'))

akPu6Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiGenJets'))

akPu6CalobTagger = bTaggers("akPu6Calo")

#create objects locally since they dont load properly otherwise
akPu6Calomatch = akPu6CalobTagger.match
akPu6Caloparton = akPu6CalobTagger.parton
akPu6CaloPatJetFlavourAssociation = akPu6CalobTagger.PatJetFlavourAssociation
akPu6CaloJetTracksAssociatorAtVertex = akPu6CalobTagger.JetTracksAssociatorAtVertex
akPu6CaloSimpleSecondaryVertexHighEffBJetTags = akPu6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu6CaloSimpleSecondaryVertexHighPurBJetTags = akPu6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu6CaloCombinedSecondaryVertexBJetTags = akPu6CalobTagger.CombinedSecondaryVertexBJetTags
akPu6CaloCombinedSecondaryVertexMVABJetTags = akPu6CalobTagger.CombinedSecondaryVertexMVABJetTags
akPu6CaloJetBProbabilityBJetTags = akPu6CalobTagger.JetBProbabilityBJetTags
akPu6CaloSoftMuonByPtBJetTags = akPu6CalobTagger.SoftMuonByPtBJetTags
akPu6CaloSoftMuonByIP3dBJetTags = akPu6CalobTagger.SoftMuonByIP3dBJetTags
akPu6CaloTrackCountingHighEffBJetTags = akPu6CalobTagger.TrackCountingHighEffBJetTags
akPu6CaloTrackCountingHighPurBJetTags = akPu6CalobTagger.TrackCountingHighPurBJetTags
akPu6CaloPatJetPartonAssociation = akPu6CalobTagger.PatJetPartonAssociation

akPu6CaloImpactParameterTagInfos = akPu6CalobTagger.ImpactParameterTagInfos
akPu6CaloJetProbabilityBJetTags = akPu6CalobTagger.JetProbabilityBJetTags
akPu6CaloPositiveOnlyJetProbabilityJetTags = akPu6CalobTagger.PositiveOnlyJetProbabilityJetTags
akPu6CaloNegativeOnlyJetProbabilityJetTags = akPu6CalobTagger.NegativeOnlyJetProbabilityJetTags
akPu6CaloNegativeTrackCountingHighEffJetTags = akPu6CalobTagger.NegativeTrackCountingHighEffJetTags
akPu6CaloNegativeTrackCountingHighPur = akPu6CalobTagger.NegativeTrackCountingHighPur
akPu6CaloNegativeOnlyJetBProbabilityJetTags = akPu6CalobTagger.NegativeOnlyJetBProbabilityJetTags
akPu6CaloPositiveOnlyJetBProbabilityJetTags = akPu6CalobTagger.PositiveOnlyJetBProbabilityJetTags

akPu6CaloSecondaryVertexTagInfos = akPu6CalobTagger.SecondaryVertexTagInfos
akPu6CaloSimpleSecondaryVertexHighEffBJetTags = akPu6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu6CaloSimpleSecondaryVertexHighPurBJetTags = akPu6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu6CaloCombinedSecondaryVertexBJetTags = akPu6CalobTagger.CombinedSecondaryVertexBJetTags
akPu6CaloCombinedSecondaryVertexMVABJetTags = akPu6CalobTagger.CombinedSecondaryVertexMVABJetTags

akPu6CaloSecondaryVertexNegativeTagInfos = akPu6CalobTagger.SecondaryVertexNegativeTagInfos
akPu6CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akPu6CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu6CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akPu6CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu6CaloCombinedSecondaryVertexNegativeBJetTags = akPu6CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akPu6CaloCombinedSecondaryVertexPositiveBJetTags = akPu6CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akPu6CaloSoftMuonTagInfos = akPu6CalobTagger.SoftMuonTagInfos
akPu6CaloSoftMuonBJetTags = akPu6CalobTagger.SoftMuonBJetTags
akPu6CaloSoftMuonByIP3dBJetTags = akPu6CalobTagger.SoftMuonByIP3dBJetTags
akPu6CaloSoftMuonByPtBJetTags = akPu6CalobTagger.SoftMuonByPtBJetTags
akPu6CaloNegativeSoftMuonByPtBJetTags = akPu6CalobTagger.NegativeSoftMuonByPtBJetTags
akPu6CaloPositiveSoftMuonByPtBJetTags = akPu6CalobTagger.PositiveSoftMuonByPtBJetTags

akPu6CaloPatJetFlavourId = cms.Sequence(akPu6CaloPatJetPartonAssociation*akPu6CaloPatJetFlavourAssociation)

akPu6CaloJetBtaggingIP       = cms.Sequence(akPu6CaloImpactParameterTagInfos *
            (akPu6CaloTrackCountingHighEffBJetTags +
             akPu6CaloTrackCountingHighPurBJetTags +
             akPu6CaloJetProbabilityBJetTags +
             akPu6CaloJetBProbabilityBJetTags +
             akPu6CaloPositiveOnlyJetProbabilityJetTags +
             akPu6CaloNegativeOnlyJetProbabilityJetTags +
             akPu6CaloNegativeTrackCountingHighEffJetTags +
             akPu6CaloNegativeTrackCountingHighPur +
             akPu6CaloNegativeOnlyJetBProbabilityJetTags +
             akPu6CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu6CaloJetBtaggingSV = cms.Sequence(akPu6CaloImpactParameterTagInfos
            *
            akPu6CaloSecondaryVertexTagInfos
            * (akPu6CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akPu6CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akPu6CaloCombinedSecondaryVertexBJetTags
                +
                akPu6CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akPu6CaloJetBtaggingNegSV = cms.Sequence(akPu6CaloImpactParameterTagInfos
            *
            akPu6CaloSecondaryVertexNegativeTagInfos
            * (akPu6CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu6CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu6CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akPu6CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu6CaloJetBtaggingMu = cms.Sequence(akPu6CaloSoftMuonTagInfos * (akPu6CaloSoftMuonBJetTags
                +
                akPu6CaloSoftMuonByIP3dBJetTags
                +
                akPu6CaloSoftMuonByPtBJetTags
                +
                akPu6CaloNegativeSoftMuonByPtBJetTags
                +
                akPu6CaloPositiveSoftMuonByPtBJetTags
              )
            )

akPu6CaloJetBtagging = cms.Sequence(akPu6CaloJetBtaggingIP
            *akPu6CaloJetBtaggingSV
            *akPu6CaloJetBtaggingNegSV
            *akPu6CaloJetBtaggingMu
            )

akPu6CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu6CaloJets"),
        genJetMatch          = cms.InputTag("akPu6Calomatch"),
        genPartonMatch       = cms.InputTag("akPu6Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu6Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu6CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu6CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu6CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu6CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu6CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu6CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu6CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu6CaloJetProbabilityBJetTags"),
            cms.InputTag("akPu6CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akPu6CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu6CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu6CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu6CaloJetID"),
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

akPu6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu6CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("hiGenParticles"),
							     eventInfoTag = cms.InputTag("hiSignal"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("akPu6Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu6CaloJetSequence_mc = cms.Sequence(
                                                  akPu6Caloclean
                                                  *
                                                  akPu6Calomatch
                                                  *
                                                  akPu6Caloparton
                                                  *
                                                  akPu6Calocorr
                                                  *
                                                  akPu6CaloJetID
                                                  *
                                                  akPu6CaloPatJetFlavourId
                                                  *
                                                  akPu6CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akPu6CaloJetBtagging
                                                  *
                                                  akPu6CalopatJetsWithBtagging
                                                  *
                                                  akPu6CaloJetAnalyzer
                                                  )

akPu6CaloJetSequence_data = cms.Sequence(akPu6Calocorr
                                                    *
                                                    akPu6CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu6CaloJetBtagging
                                                    *
                                                    akPu6CalopatJetsWithBtagging
                                                    *
                                                    akPu6CaloJetAnalyzer
                                                    )

akPu6CaloJetSequence_jec = akPu6CaloJetSequence_mc
akPu6CaloJetSequence_mix = akPu6CaloJetSequence_mc

akPu6CaloJetSequence = cms.Sequence(akPu6CaloJetSequence_mix)
