

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak6Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak6CaloJets"),
    matched = cms.InputTag("ak6GenJets"),
    maxDeltaR = 0.6
    )

ak6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak6CaloJets")
                                                        )

ak6Calocorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak6CaloJets"),
    payload = "AK6Calo_offline"
    )

ak6CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak6CaloJets'))

#ak6Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6GenJets'))

ak6CalobTagger = bTaggers("ak6Calo",0.6)

#create objects locally since they dont load properly otherwise
#ak6Calomatch = ak6CalobTagger.match
ak6Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak6CaloJets"), matched = cms.InputTag("genParticles"))
ak6CaloPatJetFlavourAssociationLegacy = ak6CalobTagger.PatJetFlavourAssociationLegacy
ak6CaloPatJetPartons = ak6CalobTagger.PatJetPartons
ak6CaloJetTracksAssociatorAtVertex = ak6CalobTagger.JetTracksAssociatorAtVertex
ak6CaloJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
ak6CaloSimpleSecondaryVertexHighEffBJetTags = ak6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak6CaloSimpleSecondaryVertexHighPurBJetTags = ak6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak6CaloCombinedSecondaryVertexBJetTags = ak6CalobTagger.CombinedSecondaryVertexBJetTags
ak6CaloCombinedSecondaryVertexV2BJetTags = ak6CalobTagger.CombinedSecondaryVertexV2BJetTags
ak6CaloJetBProbabilityBJetTags = ak6CalobTagger.JetBProbabilityBJetTags
ak6CaloSoftPFMuonByPtBJetTags = ak6CalobTagger.SoftPFMuonByPtBJetTags
ak6CaloSoftPFMuonByIP3dBJetTags = ak6CalobTagger.SoftPFMuonByIP3dBJetTags
ak6CaloTrackCountingHighEffBJetTags = ak6CalobTagger.TrackCountingHighEffBJetTags
ak6CaloTrackCountingHighPurBJetTags = ak6CalobTagger.TrackCountingHighPurBJetTags
ak6CaloPatJetPartonAssociationLegacy = ak6CalobTagger.PatJetPartonAssociationLegacy

ak6CaloImpactParameterTagInfos = ak6CalobTagger.ImpactParameterTagInfos
ak6CaloImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak6CaloJetProbabilityBJetTags = ak6CalobTagger.JetProbabilityBJetTags
ak6CaloPositiveOnlyJetProbabilityBJetTags = ak6CalobTagger.PositiveOnlyJetProbabilityBJetTags
ak6CaloNegativeOnlyJetProbabilityBJetTags = ak6CalobTagger.NegativeOnlyJetProbabilityBJetTags
ak6CaloNegativeTrackCountingHighEffBJetTags = ak6CalobTagger.NegativeTrackCountingHighEffBJetTags
ak6CaloNegativeTrackCountingHighPurBJetTags = ak6CalobTagger.NegativeTrackCountingHighPurBJetTags
ak6CaloNegativeOnlyJetBProbabilityBJetTags = ak6CalobTagger.NegativeOnlyJetBProbabilityBJetTags
ak6CaloPositiveOnlyJetBProbabilityBJetTags = ak6CalobTagger.PositiveOnlyJetBProbabilityBJetTags

ak6CaloSecondaryVertexTagInfos = ak6CalobTagger.SecondaryVertexTagInfos
ak6CaloSimpleSecondaryVertexHighEffBJetTags = ak6CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak6CaloSimpleSecondaryVertexHighPurBJetTags = ak6CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak6CaloCombinedSecondaryVertexBJetTags = ak6CalobTagger.CombinedSecondaryVertexBJetTags
ak6CaloCombinedSecondaryVertexV2BJetTags = ak6CalobTagger.CombinedSecondaryVertexV2BJetTags

ak6CaloSecondaryVertexNegativeTagInfos = ak6CalobTagger.SecondaryVertexNegativeTagInfos
ak6CaloNegativeSimpleSecondaryVertexHighEffBJetTags = ak6CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
ak6CaloNegativeSimpleSecondaryVertexHighPurBJetTags = ak6CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
ak6CaloNegativeCombinedSecondaryVertexBJetTags = ak6CalobTagger.NegativeCombinedSecondaryVertexBJetTags
ak6CaloPositiveCombinedSecondaryVertexBJetTags = ak6CalobTagger.PositiveCombinedSecondaryVertexBJetTags

ak6CaloSoftPFMuonsTagInfos = ak6CalobTagger.SoftPFMuonsTagInfos
ak6CaloSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak6CaloSoftPFMuonBJetTags = ak6CalobTagger.SoftPFMuonBJetTags
ak6CaloSoftPFMuonByIP3dBJetTags = ak6CalobTagger.SoftPFMuonByIP3dBJetTags
ak6CaloSoftPFMuonByPtBJetTags = ak6CalobTagger.SoftPFMuonByPtBJetTags
ak6CaloNegativeSoftPFMuonByPtBJetTags = ak6CalobTagger.NegativeSoftPFMuonByPtBJetTags
ak6CaloPositiveSoftPFMuonByPtBJetTags = ak6CalobTagger.PositiveSoftPFMuonByPtBJetTags
ak6CaloPatJetFlavourIdLegacy = cms.Sequence(ak6CaloPatJetPartonAssociationLegacy*ak6CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#ak6CaloPatJetFlavourAssociation = ak6CalobTagger.PatJetFlavourAssociation
#ak6CaloPatJetFlavourId = cms.Sequence(ak6CaloPatJetPartons*ak6CaloPatJetFlavourAssociation)

ak6CaloJetBtaggingIP       = cms.Sequence(ak6CaloImpactParameterTagInfos *
            (ak6CaloTrackCountingHighEffBJetTags +
             ak6CaloTrackCountingHighPurBJetTags +
             ak6CaloJetProbabilityBJetTags +
             ak6CaloJetBProbabilityBJetTags +
             ak6CaloPositiveOnlyJetProbabilityBJetTags +
             ak6CaloNegativeOnlyJetProbabilityBJetTags +
             ak6CaloNegativeTrackCountingHighEffBJetTags +
             ak6CaloNegativeTrackCountingHighPurBJetTags +
             ak6CaloNegativeOnlyJetBProbabilityBJetTags +
             ak6CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

ak6CaloJetBtaggingSV = cms.Sequence(ak6CaloImpactParameterTagInfos
            *
            ak6CaloSecondaryVertexTagInfos
            * (ak6CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak6CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak6CaloCombinedSecondaryVertexBJetTags
                +
                ak6CaloCombinedSecondaryVertexV2BJetTags
              )
            )

ak6CaloJetBtaggingNegSV = cms.Sequence(ak6CaloImpactParameterTagInfos
            *
            ak6CaloSecondaryVertexNegativeTagInfos
            * (ak6CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                ak6CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                ak6CaloNegativeCombinedSecondaryVertexBJetTags
                +
                ak6CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

ak6CaloJetBtaggingMu = cms.Sequence(ak6CaloSoftPFMuonsTagInfos * (ak6CaloSoftPFMuonBJetTags
                +
                ak6CaloSoftPFMuonByIP3dBJetTags
                +
                ak6CaloSoftPFMuonByPtBJetTags
                +
                ak6CaloNegativeSoftPFMuonByPtBJetTags
                +
                ak6CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

ak6CaloJetBtagging = cms.Sequence(ak6CaloJetBtaggingIP
            *ak6CaloJetBtaggingSV
            *ak6CaloJetBtaggingNegSV
#            *ak6CaloJetBtaggingMu
            )

ak6CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak6CaloJets"),
        genJetMatch          = cms.InputTag("ak6Calomatch"),
        genPartonMatch       = cms.InputTag("ak6Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak6Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak6CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("ak6CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak6CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("ak6CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak6CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak6CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak6CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("ak6CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak6CaloJetProbabilityBJetTags"),
            #cms.InputTag("ak6CaloSoftPFMuonByPtBJetTags"),
            #cms.InputTag("ak6CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("ak6CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak6CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak6CaloJetID"),
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

ak6CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak6CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak6GenJets',
                                                             rParam = 0.6,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
							     doSubEvent = False,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("ak6Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False)
                                                             )

ak6CaloJetSequence_mc = cms.Sequence(
                                                  #ak6Caloclean
                                                  #*
                                                  ak6Calomatch
                                                  *
                                                  ak6Caloparton
                                                  *
                                                  ak6Calocorr
                                                  *
                                                  #ak6CaloJetID
                                                  #*
                                                  ak6CaloPatJetFlavourIdLegacy
                                                  #*
			                          #ak6CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  ak6CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak6CaloJetBtagging
                                                  *
                                                  ak6CalopatJetsWithBtagging
                                                  *
                                                  ak6CaloJetAnalyzer
                                                  )

ak6CaloJetSequence_data = cms.Sequence(ak6Calocorr
                                                    *
                                                    #ak6CaloJetID
                                                    #*
                                                    ak6CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak6CaloJetBtagging
                                                    *
                                                    ak6CalopatJetsWithBtagging
                                                    *
                                                    ak6CaloJetAnalyzer
                                                    )

ak6CaloJetSequence_jec = cms.Sequence(ak6CaloJetSequence_mc)
ak6CaloJetSequence_mix = cms.Sequence(ak6CaloJetSequence_mc)

ak6CaloJetSequence = cms.Sequence(ak6CaloJetSequence_data)
