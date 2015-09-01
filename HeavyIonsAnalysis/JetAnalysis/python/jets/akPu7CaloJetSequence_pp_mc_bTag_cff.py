

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu7Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu7CaloJets"),
    matched = cms.InputTag("ak7HiGenJets"),
    maxDeltaR = 0.7
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

#akPu7Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

akPu7CalobTagger = bTaggers("akPu7Calo",0.7)

#create objects locally since they dont load properly otherwise
#akPu7Calomatch = akPu7CalobTagger.match
akPu7Caloparton = akPu7CalobTagger.parton
akPu7CaloPatJetFlavourAssociationLegacy = akPu7CalobTagger.PatJetFlavourAssociationLegacy
akPu7CaloPatJetPartons = akPu7CalobTagger.PatJetPartons
akPu7CaloJetTracksAssociatorAtVertex = akPu7CalobTagger.JetTracksAssociatorAtVertex
akPu7CaloSimpleSecondaryVertexHighEffBJetTags = akPu7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu7CaloSimpleSecondaryVertexHighPurBJetTags = akPu7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu7CaloCombinedSecondaryVertexBJetTags = akPu7CalobTagger.CombinedSecondaryVertexBJetTags
akPu7CaloCombinedSecondaryVertexV2BJetTags = akPu7CalobTagger.CombinedSecondaryVertexV2BJetTags
akPu7CaloJetBProbabilityBJetTags = akPu7CalobTagger.JetBProbabilityBJetTags
akPu7CaloSoftPFMuonByPtBJetTags = akPu7CalobTagger.SoftPFMuonByPtBJetTags
akPu7CaloSoftPFMuonByIP3dBJetTags = akPu7CalobTagger.SoftPFMuonByIP3dBJetTags
akPu7CaloTrackCountingHighEffBJetTags = akPu7CalobTagger.TrackCountingHighEffBJetTags
akPu7CaloTrackCountingHighPurBJetTags = akPu7CalobTagger.TrackCountingHighPurBJetTags
akPu7CaloPatJetPartonAssociationLegacy = akPu7CalobTagger.PatJetPartonAssociationLegacy

akPu7CaloImpactParameterTagInfos = akPu7CalobTagger.ImpactParameterTagInfos
akPu7CaloJetProbabilityBJetTags = akPu7CalobTagger.JetProbabilityBJetTags
akPu7CaloPositiveOnlyJetProbabilityBJetTags = akPu7CalobTagger.PositiveOnlyJetProbabilityBJetTags
akPu7CaloNegativeOnlyJetProbabilityBJetTags = akPu7CalobTagger.NegativeOnlyJetProbabilityBJetTags
akPu7CaloNegativeTrackCountingHighEffBJetTags = akPu7CalobTagger.NegativeTrackCountingHighEffBJetTags
akPu7CaloNegativeTrackCountingHighPurBJetTags = akPu7CalobTagger.NegativeTrackCountingHighPurBJetTags
akPu7CaloNegativeOnlyJetBProbabilityBJetTags = akPu7CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akPu7CaloPositiveOnlyJetBProbabilityBJetTags = akPu7CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akPu7CaloSecondaryVertexTagInfos = akPu7CalobTagger.SecondaryVertexTagInfos
akPu7CaloSimpleSecondaryVertexHighEffBJetTags = akPu7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu7CaloSimpleSecondaryVertexHighPurBJetTags = akPu7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu7CaloCombinedSecondaryVertexBJetTags = akPu7CalobTagger.CombinedSecondaryVertexBJetTags
akPu7CaloCombinedSecondaryVertexV2BJetTags = akPu7CalobTagger.CombinedSecondaryVertexV2BJetTags

akPu7CaloSecondaryVertexNegativeTagInfos = akPu7CalobTagger.SecondaryVertexNegativeTagInfos
akPu7CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akPu7CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akPu7CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akPu7CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akPu7CaloNegativeCombinedSecondaryVertexBJetTags = akPu7CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akPu7CaloPositiveCombinedSecondaryVertexBJetTags = akPu7CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akPu7CaloSoftPFMuonsTagInfos = akPu7CalobTagger.SoftPFMuonsTagInfos
akPu7CaloSoftPFMuonBJetTags = akPu7CalobTagger.SoftPFMuonBJetTags
akPu7CaloSoftPFMuonByIP3dBJetTags = akPu7CalobTagger.SoftPFMuonByIP3dBJetTags
akPu7CaloSoftPFMuonByPtBJetTags = akPu7CalobTagger.SoftPFMuonByPtBJetTags
akPu7CaloNegativeSoftPFMuonByPtBJetTags = akPu7CalobTagger.NegativeSoftPFMuonByPtBJetTags
akPu7CaloPositiveSoftPFMuonByPtBJetTags = akPu7CalobTagger.PositiveSoftPFMuonByPtBJetTags
akPu7CaloPatJetFlavourIdLegacy = cms.Sequence(akPu7CaloPatJetPartonAssociationLegacy*akPu7CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akPu7CaloPatJetFlavourAssociation = akPu7CalobTagger.PatJetFlavourAssociation
#akPu7CaloPatJetFlavourId = cms.Sequence(akPu7CaloPatJetPartons*akPu7CaloPatJetFlavourAssociation)

akPu7CaloJetBtaggingIP       = cms.Sequence(akPu7CaloImpactParameterTagInfos *
            (akPu7CaloTrackCountingHighEffBJetTags +
             akPu7CaloTrackCountingHighPurBJetTags +
             akPu7CaloJetProbabilityBJetTags +
             akPu7CaloJetBProbabilityBJetTags +
             akPu7CaloPositiveOnlyJetProbabilityBJetTags +
             akPu7CaloNegativeOnlyJetProbabilityBJetTags +
             akPu7CaloNegativeTrackCountingHighEffBJetTags +
             akPu7CaloNegativeTrackCountingHighPurBJetTags +
             akPu7CaloNegativeOnlyJetBProbabilityBJetTags +
             akPu7CaloPositiveOnlyJetBProbabilityBJetTags
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
                akPu7CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akPu7CaloJetBtaggingNegSV = cms.Sequence(akPu7CaloImpactParameterTagInfos
            *
            akPu7CaloSecondaryVertexNegativeTagInfos
            * (akPu7CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akPu7CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akPu7CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akPu7CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akPu7CaloJetBtaggingMu = cms.Sequence(akPu7CaloSoftPFMuonsTagInfos * (akPu7CaloSoftPFMuonBJetTags
                +
                akPu7CaloSoftPFMuonByIP3dBJetTags
                +
                akPu7CaloSoftPFMuonByPtBJetTags
                +
                akPu7CaloNegativeSoftPFMuonByPtBJetTags
                +
                akPu7CaloPositiveSoftPFMuonByPtBJetTags
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
        JetPartonMapSource   = cms.InputTag("akPu7CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akPu7CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu7CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu7CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu7CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu7CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu7CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akPu7CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu7CaloJetProbabilityBJetTags"),
            cms.InputTag("akPu7CaloSoftPFMuonByPtBJetTags"),
            cms.InputTag("akPu7CaloSoftPFMuonByIP3dBJetTags"),
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
        getJetMCFlavour = True,
        addGenPartonMatch = True,
        addGenJetMatch = True,
        embedGenJetMatch = True,
        embedGenPartonMatch = True,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

akPu7CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu7CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
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
                                                             bTagJetName = cms.untracked.string("akPu7Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

akPu7CaloJetSequence_mc = cms.Sequence(
                                                  #akPu7Caloclean
                                                  #*
                                                  akPu7Calomatch
                                                  *
                                                  akPu7Caloparton
                                                  *
                                                  akPu7Calocorr
                                                  *
                                                  akPu7CaloJetID
                                                  *
                                                  akPu7CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akPu7CaloPatJetFlavourId  # Use legacy algo till PU implemented
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

akPu7CaloJetSequence_jec = cms.Sequence(akPu7CaloJetSequence_mc)
akPu7CaloJetSequence_mix = cms.Sequence(akPu7CaloJetSequence_mc)

akPu7CaloJetSequence = cms.Sequence(akPu7CaloJetSequence_mc)
