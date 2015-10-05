

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak7Calomatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak7CaloJets"),
    matched = cms.InputTag("ak7HiGenJets"),
    maxDeltaR = 0.7
    )

ak7Caloparton = patJetPartonMatch.clone(src = cms.InputTag("ak7CaloJets")
                                                        )

ak7Calocorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak7CaloJets"),
    payload = "AK7Calo_HI"
    )

ak7CaloJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak7CaloJets'))

#ak7Caloclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

ak7CalobTagger = bTaggers("ak7Calo",0.7)

#create objects locally since they dont load properly otherwise
#ak7Calomatch = ak7CalobTagger.match
ak7Caloparton = ak7CalobTagger.parton
ak7CaloPatJetFlavourAssociationLegacy = ak7CalobTagger.PatJetFlavourAssociationLegacy
ak7CaloPatJetPartons = ak7CalobTagger.PatJetPartons
ak7CaloJetTracksAssociatorAtVertex = ak7CalobTagger.JetTracksAssociatorAtVertex
ak7CaloSimpleSecondaryVertexHighEffBJetTags = ak7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak7CaloSimpleSecondaryVertexHighPurBJetTags = ak7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak7CaloCombinedSecondaryVertexBJetTags = ak7CalobTagger.CombinedSecondaryVertexBJetTags
ak7CaloCombinedSecondaryVertexV2BJetTags = ak7CalobTagger.CombinedSecondaryVertexV2BJetTags
ak7CaloJetBProbabilityBJetTags = ak7CalobTagger.JetBProbabilityBJetTags
ak7CaloSoftPFMuonByPtBJetTags = ak7CalobTagger.SoftPFMuonByPtBJetTags
ak7CaloSoftPFMuonByIP3dBJetTags = ak7CalobTagger.SoftPFMuonByIP3dBJetTags
ak7CaloTrackCountingHighEffBJetTags = ak7CalobTagger.TrackCountingHighEffBJetTags
ak7CaloTrackCountingHighPurBJetTags = ak7CalobTagger.TrackCountingHighPurBJetTags
ak7CaloPatJetPartonAssociationLegacy = ak7CalobTagger.PatJetPartonAssociationLegacy

ak7CaloImpactParameterTagInfos = ak7CalobTagger.ImpactParameterTagInfos
ak7CaloJetProbabilityBJetTags = ak7CalobTagger.JetProbabilityBJetTags
ak7CaloPositiveOnlyJetProbabilityBJetTags = ak7CalobTagger.PositiveOnlyJetProbabilityBJetTags
ak7CaloNegativeOnlyJetProbabilityBJetTags = ak7CalobTagger.NegativeOnlyJetProbabilityBJetTags
ak7CaloNegativeTrackCountingHighEffBJetTags = ak7CalobTagger.NegativeTrackCountingHighEffBJetTags
ak7CaloNegativeTrackCountingHighPurBJetTags = ak7CalobTagger.NegativeTrackCountingHighPurBJetTags
ak7CaloNegativeOnlyJetBProbabilityBJetTags = ak7CalobTagger.NegativeOnlyJetBProbabilityBJetTags
ak7CaloPositiveOnlyJetBProbabilityBJetTags = ak7CalobTagger.PositiveOnlyJetBProbabilityBJetTags

ak7CaloSecondaryVertexTagInfos = ak7CalobTagger.SecondaryVertexTagInfos
ak7CaloSimpleSecondaryVertexHighEffBJetTags = ak7CalobTagger.SimpleSecondaryVertexHighEffBJetTags
ak7CaloSimpleSecondaryVertexHighPurBJetTags = ak7CalobTagger.SimpleSecondaryVertexHighPurBJetTags
ak7CaloCombinedSecondaryVertexBJetTags = ak7CalobTagger.CombinedSecondaryVertexBJetTags
ak7CaloCombinedSecondaryVertexV2BJetTags = ak7CalobTagger.CombinedSecondaryVertexV2BJetTags

ak7CaloSecondaryVertexNegativeTagInfos = ak7CalobTagger.SecondaryVertexNegativeTagInfos
ak7CaloNegativeSimpleSecondaryVertexHighEffBJetTags = ak7CalobTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
ak7CaloNegativeSimpleSecondaryVertexHighPurBJetTags = ak7CalobTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
ak7CaloNegativeCombinedSecondaryVertexBJetTags = ak7CalobTagger.NegativeCombinedSecondaryVertexBJetTags
ak7CaloPositiveCombinedSecondaryVertexBJetTags = ak7CalobTagger.PositiveCombinedSecondaryVertexBJetTags

ak7CaloSoftPFMuonsTagInfos = ak7CalobTagger.SoftPFMuonsTagInfos
ak7CaloSoftPFMuonBJetTags = ak7CalobTagger.SoftPFMuonBJetTags
ak7CaloSoftPFMuonByIP3dBJetTags = ak7CalobTagger.SoftPFMuonByIP3dBJetTags
ak7CaloSoftPFMuonByPtBJetTags = ak7CalobTagger.SoftPFMuonByPtBJetTags
ak7CaloNegativeSoftPFMuonByPtBJetTags = ak7CalobTagger.NegativeSoftPFMuonByPtBJetTags
ak7CaloPositiveSoftPFMuonByPtBJetTags = ak7CalobTagger.PositiveSoftPFMuonByPtBJetTags
ak7CaloPatJetFlavourIdLegacy = cms.Sequence(ak7CaloPatJetPartonAssociationLegacy*ak7CaloPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#ak7CaloPatJetFlavourAssociation = ak7CalobTagger.PatJetFlavourAssociation
#ak7CaloPatJetFlavourId = cms.Sequence(ak7CaloPatJetPartons*ak7CaloPatJetFlavourAssociation)

ak7CaloJetBtaggingIP       = cms.Sequence(ak7CaloImpactParameterTagInfos *
            (ak7CaloTrackCountingHighEffBJetTags +
             ak7CaloTrackCountingHighPurBJetTags +
             ak7CaloJetProbabilityBJetTags +
             ak7CaloJetBProbabilityBJetTags +
             ak7CaloPositiveOnlyJetProbabilityBJetTags +
             ak7CaloNegativeOnlyJetProbabilityBJetTags +
             ak7CaloNegativeTrackCountingHighEffBJetTags +
             ak7CaloNegativeTrackCountingHighPurBJetTags +
             ak7CaloNegativeOnlyJetBProbabilityBJetTags +
             ak7CaloPositiveOnlyJetBProbabilityBJetTags
            )
            )

ak7CaloJetBtaggingSV = cms.Sequence(ak7CaloImpactParameterTagInfos
            *
            ak7CaloSecondaryVertexTagInfos
            * (ak7CaloSimpleSecondaryVertexHighEffBJetTags
                +
                ak7CaloSimpleSecondaryVertexHighPurBJetTags
                +
                ak7CaloCombinedSecondaryVertexBJetTags
                +
                ak7CaloCombinedSecondaryVertexV2BJetTags
              )
            )

ak7CaloJetBtaggingNegSV = cms.Sequence(ak7CaloImpactParameterTagInfos
            *
            ak7CaloSecondaryVertexNegativeTagInfos
            * (ak7CaloNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                ak7CaloNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                ak7CaloNegativeCombinedSecondaryVertexBJetTags
                +
                ak7CaloPositiveCombinedSecondaryVertexBJetTags
              )
            )

ak7CaloJetBtaggingMu = cms.Sequence(ak7CaloSoftPFMuonsTagInfos * (ak7CaloSoftPFMuonBJetTags
                +
                ak7CaloSoftPFMuonByIP3dBJetTags
                +
                ak7CaloSoftPFMuonByPtBJetTags
                +
                ak7CaloNegativeSoftPFMuonByPtBJetTags
                +
                ak7CaloPositiveSoftPFMuonByPtBJetTags
              )
            )

ak7CaloJetBtagging = cms.Sequence(ak7CaloJetBtaggingIP
            *ak7CaloJetBtaggingSV
            *ak7CaloJetBtaggingNegSV
            *ak7CaloJetBtaggingMu
            )

ak7CalopatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak7CaloJets"),
        genJetMatch          = cms.InputTag("ak7Calomatch"),
        genPartonMatch       = cms.InputTag("ak7Caloparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak7Calocorr")),
        JetPartonMapSource   = cms.InputTag("ak7CaloPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("ak7CaloPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak7CaloJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("ak7CaloSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak7CaloSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak7CaloCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak7CaloCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("ak7CaloJetBProbabilityBJetTags"),
            cms.InputTag("ak7CaloJetProbabilityBJetTags"),
            cms.InputTag("ak7CaloSoftPFMuonByPtBJetTags"),
            cms.InputTag("ak7CaloSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("ak7CaloTrackCountingHighEffBJetTags"),
            cms.InputTag("ak7CaloTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak7CaloJetID"),
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

ak7CaloJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak7CalopatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
                                                             genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(True),
                                                             bTagJetName = cms.untracked.string("ak7Calo"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak7CaloJetSequence_mc = cms.Sequence(
                                                  #ak7Caloclean
                                                  #*
                                                  ak7Calomatch
                                                  *
                                                  ak7Caloparton
                                                  *
                                                  ak7Calocorr
                                                  *
                                                  ak7CaloJetID
                                                  *
                                                  ak7CaloPatJetFlavourIdLegacy
                                                  #*
			                          #ak7CaloPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  ak7CaloJetTracksAssociatorAtVertex
                                                  *
                                                  ak7CaloJetBtagging
                                                  *
                                                  ak7CalopatJetsWithBtagging
                                                  *
                                                  ak7CaloJetAnalyzer
                                                  )

ak7CaloJetSequence_data = cms.Sequence(ak7Calocorr
                                                    *
                                                    ak7CaloJetTracksAssociatorAtVertex
                                                    *
                                                    ak7CaloJetBtagging
                                                    *
                                                    ak7CalopatJetsWithBtagging
                                                    *
                                                    ak7CaloJetAnalyzer
                                                    )

ak7CaloJetSequence_jec = cms.Sequence(ak7CaloJetSequence_mc)
ak7CaloJetSequence_mix = cms.Sequence(ak7CaloJetSequence_mc)

ak7CaloJetSequence = cms.Sequence(ak7CaloJetSequence_mc)
