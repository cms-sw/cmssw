

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak2Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak2CaloJets"),
    matched = cms.InputTag("ak2HiGenJets"),
    maxDeltaR = 0.2
    )

ak2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak2CaloJets")
                                                        )

ak2Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak2CaloJets"),
    payload = "AK2Calo_offline"
    )

ak2CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak2CaloJets'))

#ak2Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJets'))

ak2CalobTagger = bTaggers("ak2Calo",0.2)

#create objects locally since they dont load properly otherwise
#ak2Calomatch = ak2CalobTagger.match
ak2Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak2CaloJets"), matched = cms.InputTag("genParticles"))
ak2CaloPatJetFlavourAssociationLegacy = ak2CalobTagger.PatJetFlavourAssociationLegacy
ak2CaloPatJetPartons = ak2CalobTagger.PatJetPartons
ak2CaloJetTracksAssociatorAtVertex = ak2CalobTagger.JetTracksAssociatorAtVertex
ak2CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
ak2CaloSimpleSecondaryVertexHighEffBJetTags = ak2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak2CaloSimpleSecondaryVertexHighPurBJetTags = ak2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak2CaloCombinedSecondaryVertexBJetTags = ak2CalobTagger.CombinedSecondaryVertexBJetTags
ak2CaloCombinedSecondaryVertexV2BJetTags = ak2CalobTagger.CombinedSecondaryVertexV2BJetTags
ak2CaloJetBProbabilityBJetTags = ak2CalobTagger.JetBProbabilityBJetTags
ak2CaloSoftPFMuonByPtBJetTags = ak2CalobTagger.SoftPFMuonByPtBJetTags
ak2CaloSoftPFMuonByIP3dBJetTags = ak2CalobTagger.SoftPFMuonByIP3dBJetTags
ak2CaloTrackCountingHighEffBJetTags = ak2CalobTagger.TrackCountingHighEffBJetTags
ak2CaloTrackCountingHighPurBJetTags = ak2CalobTagger.TrackCountingHighPurBJetTags
ak2CaloPatJetPartonAssociationLegacy = ak2CalobTagger.PatJetPartonAssociationLegacy

ak2CaloImpactParameterTagInfos = ak2CalobTagger.ImpactParameterTagInfos
ak2CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak2CaloJetProbabilityBJetTags = ak2CalobTagger.JetProbabilityBJetTags
ak2CaloPositiveOnlyJetProbabilityBJetTags = ak2CalobTagger.PositiveOnlyJetProbabilityBJetTags
ak2CaloNegativeOnlyJetProbabilityBJetTags = ak2CalobTagger.NegativeOnlyJetProbabilityBJetTags
ak2CaloNegativeTrackCountingHighEffBJetTags = ak2CalobTagger.NegativeTrackCountingHighEffBJetTags
ak2CaloNegativeTrackCountingHighPurBJetTags = ak2CalobTagger.NegativeTrackCountingHighPurBJetTags
ak2CaloNegativeOnlyJetBProbabilityBJetTags = ak2CalobTagger.NegativeOnlyJetBProbabilityBJetTags
ak2CaloPositiveOnlyJetBProbabilityBJetTags = ak2CalobTagger.PositiveOnlyJetBProbabilityBJetTags

ak2CaloSecondaryVertexTagInfos = ak2CalobTagger.SecondaryVertexTagInfos
ak2CaloSimpleSecondaryVertexHighEffBJetTags = ak2CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak2CaloSimpleSecondaryVertexHighPurBJetTags = ak2CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak2CaloCombinedSecondaryVertexBJetTags = ak2CalobTagger.CombinedSecondaryVertexBJetTags
ak2CaloCombinedSecondaryVertexV2BJetTags = ak2CalobTagger.CombinedSecondaryVertexV2BJetTags

ak2CaloSecondaryVertexNegativeTagInfos = ak2CalobTagger.SecondaryVertexNegativeTagInfos
ak2CaloNegativeSimpleSecondaryVertexHighEffBJetTags = ak2CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
ak2CaloNegativeSimpleSecondaryVertexHighPurBJetTags = ak2CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
ak2CaloNegativeCombinedSecondaryVertexBJetTags = ak2CalobTagger.NegativeCombinedSecondaryVertexBJetTags
ak2CaloPositiveCombinedSecondaryVertexBJetTags = ak2CalobTagger.PositiveCombinedSecondaryVertexBJetTags

ak2CaloSoftPFMuonsTagInfos = ak2CalobTagger.SoftPFMuonsTagInfos
ak2CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak2CaloSoftPFMuonBJetTags = ak2CalobTagger.SoftPFMuonBJetTags
ak2CaloSoftPFMuonByIP3dBJetTags = ak2CalobTagger.SoftPFMuonByIP3dBJetTags
ak2CaloSoftPFMuonByPtBJetTags = ak2CalobTagger.SoftPFMuonByPtBJetTags
ak2CaloNegativeSoftPFMuonByPtBJetTags = ak2CalobTagger.NegativeSoftPFMuonByPtBJetTags
ak2CaloPositiveSoftPFMuonByPtBJetTags = ak2CalobTagger.PositiveSoftPFMuonByPtBJetTags
ak2CaloPatJetFlavourIdLegacy = cms.Sequence(ak2CaloPatJetPartonAssociationLegacy*ak2CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#ak2CaloPatJetFlavourAssociation = ak2CalobTagger.PatJetFlavourAssociation
#ak2CaloPatJetFlavourId = cms.Sequence(ak2CaloPatJetPartons*ak2CaloPatJetFlavourAssociation)

ak2CaloJetBtaggingIP       = cms.Sequence(ak2CaloImpactParameterTagInfos *
            (ak2CaloTrackCountingHighEffBJetTags +
             ak2CaloTrackCountingHighPurBJetTags +
             ak2CaloJetProbabilityBJetTags +
             ak2CaloJetBProbabilityBJetTags +
             ak2CaloPositiveOnlyJetProbabilityBJetTags +
             ak2CaloNegativeOnlyJetProbabilityBJetTags +
             ak2CaloNegativeTrackCountingHighEffBJetTags +
             ak2CaloNegativeTrackCountingHighPurBJetTags +
             ak2CaloNegativeOnlyJetBProbabilityBJetTags +
             ak2CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

ak2CaloJetBtaggingSV = cms.Sequence(ak2CaloImpactParameterTagInfos
            *
            ak2CaloSecondaryVertexTagInfos
            * (ak2CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak2CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak2CaloCombinedSecondaryVertexBJetTags
                +
                ak2CaloCombinedSecondaryVertexV2BJetTags
              )
            )

ak2CaloJetBtaggingNegSV = cms.Sequence(ak2CaloImpactParameterTagInfos
            *
            ak2CaloSecondaryVertexNegativeTagInfos
            * (ak2CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                ak2CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                ak2CaloNegativeCombinedSecondaryVertexBJetTags
                +
                ak2CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

ak2CaloJetBtaggingMu = cms.Sequence(ak2CaloSoftPFMuonsTagInfos * (ak2CaloSoftPFMuonBJetTags
                +
                ak2CaloSoftPFMuonByIP3dBJetTags
                +
                ak2CaloSoftPFMuonByPtBJetTags
                +
                ak2CaloNegativeSoftPFMuonByPtBJetTags
                +
                ak2CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

ak2CaloJetBtagging = cms.Sequence(ak2CaloJetBtaggingIP
            *ak2CaloJetBtaggingSV
            *ak2CaloJetBtaggingNegSV
#            *ak2CaloJetBtaggingMu
            )

ak2CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak2CaloJets"),
        genJetMatch          = cms.InputTag("ak2Calomatch"),
        genPartonMatch       = cms.InputTag("ak2Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak2Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak2CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("ak2CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak2CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("ak2CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak2CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak2CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak2CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("ak2CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak2CaloJetProbabilityBJetTags"),
            #cms.InputTag("ak2CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("ak2CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("ak2CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak2CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak2CaloJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = False,
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

ak2CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak2CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak2HiGenJets',
                                                             rParam = 0.2,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
							     doSubEvent = False,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("ak2Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

ak2CaloJetSequence_mc = cms.Sequence(
                                                  #ak2Caloclean
                                                  #*
                                                  ak2Calomatch
                                                  *
                                                  ak2Caloparton
                                                  *
                                                  ak2Calocorr
                                                  *
                                                  #ak2CaloJetID
                                                  #*
                                                  ak2CaloPatJetFlavourIdLegacy
                                                  #*
			                          #ak2CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  ak2CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak2CaloJetBtagging
                                                  *
                                                  ak2CalopatJetsWithBtagging
                                                  *
                                                  ak2CaloJetAnalyzer
                                                  )

ak2CaloJetSequence_data = cms.Sequence(ak2Calocorr
                                                    *
                                                    #ak2CaloJetID
                                                    #*
                                                    ak2CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak2CaloJetBtagging
                                                    *
                                                    ak2CalopatJetsWithBtagging
                                                    *
                                                    ak2CaloJetAnalyzer
                                                    )

ak2CaloJetSequence_jec = cms.Sequence(ak2CaloJetSequence_mc)
ak2CaloJetSequence_mix = cms.Sequence(ak2CaloJetSequence_mc)

ak2CaloJetSequence = cms.Sequence(ak2CaloJetSequence_data)
