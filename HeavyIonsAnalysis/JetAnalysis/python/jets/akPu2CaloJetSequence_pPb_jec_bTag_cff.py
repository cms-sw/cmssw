

import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.patHeavyIonSequences_cff import *
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu2CaloJets"),
    matched = cms.InputTag("ak2HiGenJets")
    )

akPu2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu2CaloJets")
                                                        )

akPu2Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu2CaloJets"),
    payload = "AKPu2Calo_HI"
    )

akPu2CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu2CaloJets'))

akPu2Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJets'))

akPu2CalobTagger = bTaggers("akPu2Calo")

#create objects locally since they dont load properly otherwise
akPu2Calomatch = akPu2CalobTagger.match
akPu2Caloparton = akPu2CalobTagger.parton
akPu2CaloPatJetFlavourAssociation = akPu2CalobTagger.PatJetFlavourAssociation
akPu2CaloJetTracksAssociatorAtVertex = akPu2CalobTagger.JetTracksAssociatorAtVertex
akPu2CaloSimpleSecondaryVertexHighEffBJetTags = akPu2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu2CaloSimpleSecondaryVertexHighPurBJetTags = akPu2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu2CaloCombinedSecondaryVertexBJetTags = akPu2CalobTagger.CombinedSecondaryVertexBJetTags
akPu2CaloCombinedSecondaryVertexMVABJetTags = akPu2CalobTagger.CombinedSecondaryVertexMVABJetTags
akPu2CaloJetBProbabilityBJetTags = akPu2CalobTagger.JetBProbabilityBJetTags
akPu2CaloSoftMuonByPtBJetTags = akPu2CalobTagger.SoftMuonByPtBJetTags
akPu2CaloSoftMuonByIP3dBJetTags = akPu2CalobTagger.SoftMuonByIP3dBJetTags
akPu2CaloTrackCountingHighEffBJetTags = akPu2CalobTagger.TrackCountingHighEffBJetTags
akPu2CaloTrackCountingHighPurBJetTags = akPu2CalobTagger.TrackCountingHighPurBJetTags
akPu2CaloPatJetPartonAssociation = akPu2CalobTagger.PatJetPartonAssociation

akPu2CaloImpactParameterTagInfos = akPu2CalobTagger.ImpactParameterTagInfos
akPu2CaloJetProbabilityBJetTags = akPu2CalobTagger.JetProbabilityBJetTags
akPu2CaloPositiveOnlyJetProbabilityJetTags = akPu2CalobTagger.PositiveOnlyJetProbabilityJetTags
akPu2CaloNegativeOnlyJetProbabilityJetTags = akPu2CalobTagger.NegativeOnlyJetProbabilityJetTags
akPu2CaloNegativeTrackCountingHighEffJetTags = akPu2CalobTagger.NegativeTrackCountingHighEffJetTags
akPu2CaloNegativeTrackCountingHighPur = akPu2CalobTagger.NegativeTrackCountingHighPur
akPu2CaloNegativeOnlyJetBProbabilityJetTags = akPu2CalobTagger.NegativeOnlyJetBProbabilityJetTags
akPu2CaloPositiveOnlyJetBProbabilityJetTags = akPu2CalobTagger.PositiveOnlyJetBProbabilityJetTags

akPu2CaloSecondaryVertexTagInfos = akPu2CalobTagger.SecondaryVertexTagInfos
akPu2CaloSimpleSecondaryVertexHighEffBJetTags = akPu2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu2CaloSimpleSecondaryVertexHighPurBJetTags = akPu2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu2CaloCombinedSecondaryVertexBJetTags = akPu2CalobTagger.CombinedSecondaryVertexBJetTags
akPu2CaloCombinedSecondaryVertexMVABJetTags = akPu2CalobTagger.CombinedSecondaryVertexMVABJetTags

akPu2CaloSecondaryVertexNegativeTagInfos = akPu2CalobTagger.SecondaryVertexNegativeTagInfos
akPu2CaloSimpleSecondaryVertexNegativeHighEffBJetTags = akPu2CalobTagger.SimpleSecondaryVertexNegativeHighEffBJetTags
akPu2CaloSimpleSecondaryVertexNegativeHighPurBJetTags = akPu2CalobTagger.SimpleSecondaryVertexNegativeHighPurBJetTags
akPu2CaloCombinedSecondaryVertexNegativeBJetTags = akPu2CalobTagger.CombinedSecondaryVertexNegativeBJetTags
akPu2CaloCombinedSecondaryVertexPositiveBJetTags = akPu2CalobTagger.CombinedSecondaryVertexPositiveBJetTags

akPu2CaloSoftMuonTagInfos = akPu2CalobTagger.SoftMuonTagInfos
akPu2CaloSoftMuonBJetTags = akPu2CalobTagger.SoftMuonBJetTags
akPu2CaloSoftMuonByIP3dBJetTags = akPu2CalobTagger.SoftMuonByIP3dBJetTags
akPu2CaloSoftMuonByPtBJetTags = akPu2CalobTagger.SoftMuonByPtBJetTags
akPu2CaloNegativeSoftMuonByPtBJetTags = akPu2CalobTagger.NegativeSoftMuonByPtBJetTags
akPu2CaloPositiveSoftMuonByPtBJetTags = akPu2CalobTagger.PositiveSoftMuonByPtBJetTags

akPu2CaloPatJetFlavourId = cms.Sequence(akPu2CaloPatJetPartonAssociation*akPu2CaloPatJetFlavourAssociation)

akPu2CaloJetBtaggingIP       = cms.Sequence(akPu2CaloImpactParameterTagInfos *
            (akPu2CaloTrackCountingHighEffBJetTags +
             akPu2CaloTrackCountingHighPurBJetTags +
             akPu2CaloJetProbabilityBJetTags +
             akPu2CaloJetBProbabilityBJetTags +
             akPu2CaloPositiveOnlyJetProbabilityJetTags +
             akPu2CaloNegativeOnlyJetProbabilityJetTags +
             akPu2CaloNegativeTrackCountingHighEffJetTags +
             akPu2CaloNegativeTrackCountingHighPur +
             akPu2CaloNegativeOnlyJetBProbabilityJetTags +
             akPu2CaloPositiveOnlyJetBProbabilityJetTags
            )
            )

akPu2CaloJetBtaggingSV = cms.Sequence(akPu2CaloImpactParameterTagInfos
            *
            akPu2CaloSecondaryVertexTagInfos
            * (akPu2CaloSimpleSecondaryVertexHighEffBJetTags
                +
                akPu2CaloSimpleSecondaryVertexHighPurBJetTags
                +
                akPu2CaloCombinedSecondaryVertexBJetTags
                +
                akPu2CaloCombinedSecondaryVertexMVABJetTags
              )
            )

akPu2CaloJetBtaggingNegSV = cms.Sequence(akPu2CaloImpactParameterTagInfos
            *
            akPu2CaloSecondaryVertexNegativeTagInfos
            * (akPu2CaloSimpleSecondaryVertexNegativeHighEffBJetTags
                +
                akPu2CaloSimpleSecondaryVertexNegativeHighPurBJetTags
                +
                akPu2CaloCombinedSecondaryVertexNegativeBJetTags
                +
                akPu2CaloCombinedSecondaryVertexPositiveBJetTags
              )
            )

akPu2CaloJetBtaggingMu = cms.Sequence(akPu2CaloSoftMuonTagInfos * (akPu2CaloSoftMuonBJetTags
                +
                akPu2CaloSoftMuonByIP3dBJetTags
                +
                akPu2CaloSoftMuonByPtBJetTags
                +
                akPu2CaloNegativeSoftMuonByPtBJetTags
                +
                akPu2CaloPositiveSoftMuonByPtBJetTags
              )
            )

akPu2CaloJetBtagging = cms.Sequence(akPu2CaloJetBtaggingIP
            *akPu2CaloJetBtaggingSV
            *akPu2CaloJetBtaggingNegSV
            *akPu2CaloJetBtaggingMu
            )

akPu2CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu2CaloJets"),
        genJetMatch          = cms.InputTag("akPu2Calomatch"),
        genPartonMatch       = cms.InputTag("akPu2Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu2Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu2CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu2CaloJetTracksAssociatorAtVertex"),
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu2CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu2CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu2CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu2CaloCombinedSecondaryVertexMVABJetTags"),
            cms.InputTag("akPu2CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu2CaloJetProbabilityBJetTags"),
            cms.InputTag("akPu2CaloSoftMuonByPtBJetTags"),
            cms.InputTag("akPu2CaloSoftMuonByIP3dBJetTags"),
            cms.InputTag("akPu2CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu2CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu2CaloJetID"),
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

akPu2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu2CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
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
                                                             bTagJetName = cms.untracked.string("akPu2Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu2CaloJetSequence_mc = cms.Sequence(
                                                  akPu2Caloclean
                                                  *
                                                  akPu2Calomatch
                                                  *
                                                  akPu2Caloparton
                                                  *
                                                  akPu2Calocorr
                                                  *
                                                  akPu2CaloJetID
                                                  *
                                                  akPu2CaloPatJetFlavourId
                                                  *
                                                  akPu2CaloJetTracksAssociatorAtVertex
                                                  *
                                                  akPu2CaloJetBtagging
                                                  *
                                                  akPu2CalopatJetsWithBtagging
                                                  *
                                                  akPu2CaloJetAnalyzer
                                                  )

akPu2CaloJetSequence_data = cms.Sequence(akPu2Calocorr
                                                    *
                                                    akPu2CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu2CaloJetBtagging
                                                    *
                                                    akPu2CalopatJetsWithBtagging
                                                    *
                                                    akPu2CaloJetAnalyzer
                                                    )

akPu2CaloJetSequence_jec = akPu2CaloJetSequence_mc
akPu2CaloJetSequence_mix = akPu2CaloJetSequence_mc

akPu2CaloJetSequence = cms.Sequence(akPu2CaloJetSequence_jec)
akPu2CaloJetAnalyzer.genPtMin = cms.untracked.double(1)
