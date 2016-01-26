

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu2CaloJets"),
    matched = cms.InputTag("ak2HiGenJets"),
    maxDeltaR = 0.2
    )

akPu2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu2CaloJets")
                                                        )

akPu2Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu2CaloJets"),
    payload = "AKPu2Calo_offline"
    )

akPu2CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu2CaloJets'))

#akPu2Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJets'))

akPu2CalobTagger = bTaggers("akPu2Calo",0.2)

#create objects locally since they dont load properly otherwise
#akPu2Calomatch = akPu2CalobTagger.match
akPu2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("akPu2CaloJets"), matched = cms.InputTag("genParticles"))
akPu2CaloPatJetFlavourAssociationLegacy = akPu2CalobTagger.PatJetFlavourAssociationLegacy
akPu2CaloPatJetPartons = akPu2CalobTagger.PatJetPartons
akPu2CaloJetTracksAssociatorAtVertex = akPu2CalobTagger.JetTracksAssociatorAtVertex
akPu2CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akPu2CaloSimpleSecondaryVertexHighEffBJetTags = akPu2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu2CaloSimpleSecondaryVertexHighPurBJetTags = akPu2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu2CaloCombinedSecondaryVertexBJetTags = akPu2CalobTagger.CombinedSecondaryVertexBJetTags
akPu2CaloCombinedSecondaryVertexV2BJetTags = akPu2CalobTagger.CombinedSecondaryVertexV2BJetTags
akPu2CaloJetBProbabilityBJetTags = akPu2CalobTagger.JetBProbabilityBJetTags
akPu2CaloSoftPFMuonByPtBJetTags = akPu2CalobTagger.SoftPFMuonByPtBJetTags
akPu2CaloSoftPFMuonByIP3dBJetTags = akPu2CalobTagger.SoftPFMuonByIP3dBJetTags
akPu2CaloTrackCountingHighEffBJetTags = akPu2CalobTagger.TrackCountingHighEffBJetTags
akPu2CaloTrackCountingHighPurBJetTags = akPu2CalobTagger.TrackCountingHighPurBJetTags
akPu2CaloPatJetPartonAssociationLegacy = akPu2CalobTagger.PatJetPartonAssociationLegacy

akPu2CaloImpactParameterTagInfos = akPu2CalobTagger.ImpactParameterTagInfos
akPu2CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu2CaloJetProbabilityBJetTags = akPu2CalobTagger.JetProbabilityBJetTags
akPu2CaloPositiveOnlyJetProbabilityBJetTags = akPu2CalobTagger.PositiveOnlyJetProbabilityBJetTags
akPu2CaloNegativeOnlyJetProbabilityBJetTags = akPu2CalobTagger.NegativeOnlyJetProbabilityBJetTags
akPu2CaloNegativeTrackCountingHighEffBJetTags = akPu2CalobTagger.NegativeTrackCountingHighEffBJetTags
akPu2CaloNegativeTrackCountingHighPurBJetTags = akPu2CalobTagger.NegativeTrackCountingHighPurBJetTags
akPu2CaloNegativeOnlyJetBProbabilityBJetTags = akPu2CalobTagger.NegativeOnlyJetBProbabilityBJetTags
akPu2CaloPositiveOnlyJetBProbabilityBJetTags = akPu2CalobTagger.PositiveOnlyJetBProbabilityBJetTags

akPu2CaloSecondaryVertexTagInfos = akPu2CalobTagger.SecondaryVertexTagInfos
akPu2CaloSimpleSecondaryVertexHighEffBJetTags = akPu2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
akPu2CaloSimpleSecondaryVertexHighPurBJetTags = akPu2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
akPu2CaloCombinedSecondaryVertexBJetTags = akPu2CalobTagger.CombinedSecondaryVertexBJetTags
akPu2CaloCombinedSecondaryVertexV2BJetTags = akPu2CalobTagger.CombinedSecondaryVertexV2BJetTags

akPu2CaloSecondaryVertexNegativeTagInfos = akPu2CalobTagger.SecondaryVertexNegativeTagInfos
akPu2CaloNegativeSimpleSecondaryVertexHighEffBJetTags = akPu2CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akPu2CaloNegativeSimpleSecondaryVertexHighPurBJetTags = akPu2CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akPu2CaloNegativeCombinedSecondaryVertexBJetTags = akPu2CalobTagger.NegativeCombinedSecondaryVertexBJetTags
akPu2CaloPositiveCombinedSecondaryVertexBJetTags = akPu2CalobTagger.PositiveCombinedSecondaryVertexBJetTags

akPu2CaloSoftPFMuonsTagInfos = akPu2CalobTagger.SoftPFMuonsTagInfos
akPu2CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu2CaloSoftPFMuonBJetTags = akPu2CalobTagger.SoftPFMuonBJetTags
akPu2CaloSoftPFMuonByIP3dBJetTags = akPu2CalobTagger.SoftPFMuonByIP3dBJetTags
akPu2CaloSoftPFMuonByPtBJetTags = akPu2CalobTagger.SoftPFMuonByPtBJetTags
akPu2CaloNegativeSoftPFMuonByPtBJetTags = akPu2CalobTagger.NegativeSoftPFMuonByPtBJetTags
akPu2CaloPositiveSoftPFMuonByPtBJetTags = akPu2CalobTagger.PositiveSoftPFMuonByPtBJetTags
akPu2CaloPatJetFlavourIdLegacy = cms.Sequence(akPu2CaloPatJetPartonAssociationLegacy*akPu2CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akPu2CaloPatJetFlavourAssociation = akPu2CalobTagger.PatJetFlavourAssociation
#akPu2CaloPatJetFlavourId = cms.Sequence(akPu2CaloPatJetPartons*akPu2CaloPatJetFlavourAssociation)

akPu2CaloJetBtaggingIP       = cms.Sequence(akPu2CaloImpactParameterTagInfos *
            (akPu2CaloTrackCountingHighEffBJetTags +
             akPu2CaloTrackCountingHighPurBJetTags +
             akPu2CaloJetProbabilityBJetTags +
             akPu2CaloJetBProbabilityBJetTags +
             akPu2CaloPositiveOnlyJetProbabilityBJetTags +
             akPu2CaloNegativeOnlyJetProbabilityBJetTags +
             akPu2CaloNegativeTrackCountingHighEffBJetTags +
             akPu2CaloNegativeTrackCountingHighPurBJetTags +
             akPu2CaloNegativeOnlyJetBProbabilityBJetTags +
             akPu2CaloPositiveOnlyJetBProbabilityBJetTags
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
                akPu2CaloCombinedSecondaryVertexV2BJetTags
              )
            )

akPu2CaloJetBtaggingNegSV = cms.Sequence(akPu2CaloImpactParameterTagInfos
            *
            akPu2CaloSecondaryVertexNegativeTagInfos
            * (akPu2CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akPu2CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akPu2CaloNegativeCombinedSecondaryVertexBJetTags
                +
                akPu2CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

akPu2CaloJetBtaggingMu = cms.Sequence(akPu2CaloSoftPFMuonsTagInfos * (akPu2CaloSoftPFMuonBJetTags
                +
                akPu2CaloSoftPFMuonByIP3dBJetTags
                +
                akPu2CaloSoftPFMuonByPtBJetTags
                +
                akPu2CaloNegativeSoftPFMuonByPtBJetTags
                +
                akPu2CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

akPu2CaloJetBtagging = cms.Sequence(akPu2CaloJetBtaggingIP
            *akPu2CaloJetBtaggingSV
            *akPu2CaloJetBtaggingNegSV
#            *akPu2CaloJetBtaggingMu
            )

akPu2CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu2CaloJets"),
        genJetMatch          = cms.InputTag("akPu2Calomatch"),
        genPartonMatch       = cms.InputTag("akPu2Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu2Calocorr")),
        JetPartonMapSource   = cms.InputTag("akPu2CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akPu2CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu2CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu2CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu2CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu2CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu2CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akPu2CaloJetBProbabilityBJetTags"),
            cms.InputTag("akPu2CaloJetProbabilityBJetTags"),
            #cms.InputTag("akPu2CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akPu2CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akPu2CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu2CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu2CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = False,
        getJetMCFlavour = True,
        addGenPartonMatch = True,
        addGenJetMatch = True,
        embedGenJetMatch = True,
        embedGenPartonMatch = True,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

akPu2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu2CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
							     doSubEvent = True,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("akPu2Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akPu2CaloJetSequence_mc = cms.Sequence(
                                                  #akPu2Caloclean
                                                  #*
                                                  akPu2Calomatch
                                                  *
                                                  akPu2Caloparton
                                                  *
                                                  akPu2Calocorr
                                                  *
                                                  #akPu2CaloJetID
                                                  #*
                                                  akPu2CaloPatJetFlavourIdLegacy
                                                  #*
			                          #akPu2CaloPatJetFlavourId  # Use legacy algo till PU implemented
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
                                                    #akPu2CaloJetID
                                                    #*
                                                    akPu2CaloJetTracksAssociatorAtVertex
                                                    *
                                                    akPu2CaloJetBtagging
                                                    *
                                                    akPu2CalopatJetsWithBtagging
                                                    *
                                                    akPu2CaloJetAnalyzer
                                                    )

akPu2CaloJetSequence_jec = cms.Sequence(akPu2CaloJetSequence_mc)
akPu2CaloJetSequence_mix = cms.Sequence(akPu2CaloJetSequence_mc)

akPu2CaloJetSequence = cms.Sequence(akPu2CaloJetSequence_jec)
akPu2CaloJetAnalyzer.genPtMin = cms.untracked.double(1)
akPu2CaloJetAnalyzer.jetPtMin = cms.untracked.double(1)
