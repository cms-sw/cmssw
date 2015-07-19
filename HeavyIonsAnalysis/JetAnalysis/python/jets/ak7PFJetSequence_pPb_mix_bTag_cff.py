

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak7PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak7PFJets"),
    matched = cms.InputTag("ak7HiGenJets"),
    maxDeltaR = 0.7
    )

ak7PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak7PFJets")
                                                        )

ak7PFcorr = patJetCorrFactors.clone(
    useNPV = False,
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak7PFJets"),
    payload = "AK7PF_generalTracks"
    )

ak7PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak7CaloJets'))

#ak7PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak7HiGenJets'))

ak7PFbTagger = bTaggers("ak7PF",0.7)

#create objects locally since they dont load properly otherwise
#ak7PFmatch = ak7PFbTagger.match
ak7PFparton = ak7PFbTagger.parton
ak7PFPatJetFlavourAssociationLegacy = ak7PFbTagger.PatJetFlavourAssociationLegacy
ak7PFPatJetPartons = ak7PFbTagger.PatJetPartons
ak7PFJetTracksAssociatorAtVertex = ak7PFbTagger.JetTracksAssociatorAtVertex
ak7PFSimpleSecondaryVertexHighEffBJetTags = ak7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak7PFSimpleSecondaryVertexHighPurBJetTags = ak7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak7PFCombinedSecondaryVertexBJetTags = ak7PFbTagger.CombinedSecondaryVertexBJetTags
ak7PFCombinedSecondaryVertexV2BJetTags = ak7PFbTagger.CombinedSecondaryVertexV2BJetTags
ak7PFJetBProbabilityBJetTags = ak7PFbTagger.JetBProbabilityBJetTags
ak7PFSoftPFMuonByPtBJetTags = ak7PFbTagger.SoftPFMuonByPtBJetTags
ak7PFSoftPFMuonByIP3dBJetTags = ak7PFbTagger.SoftPFMuonByIP3dBJetTags
ak7PFTrackCountingHighEffBJetTags = ak7PFbTagger.TrackCountingHighEffBJetTags
ak7PFTrackCountingHighPurBJetTags = ak7PFbTagger.TrackCountingHighPurBJetTags
ak7PFPatJetPartonAssociationLegacy = ak7PFbTagger.PatJetPartonAssociationLegacy

ak7PFImpactParameterTagInfos = ak7PFbTagger.ImpactParameterTagInfos
ak7PFJetProbabilityBJetTags = ak7PFbTagger.JetProbabilityBJetTags
ak7PFPositiveOnlyJetProbabilityBJetTags = ak7PFbTagger.PositiveOnlyJetProbabilityBJetTags
ak7PFNegativeOnlyJetProbabilityBJetTags = ak7PFbTagger.NegativeOnlyJetProbabilityBJetTags
ak7PFNegativeTrackCountingHighEffBJetTags = ak7PFbTagger.NegativeTrackCountingHighEffBJetTags
ak7PFNegativeTrackCountingHighPurBJetTags = ak7PFbTagger.NegativeTrackCountingHighPurBJetTags
ak7PFNegativeOnlyJetBProbabilityBJetTags = ak7PFbTagger.NegativeOnlyJetBProbabilityBJetTags
ak7PFPositiveOnlyJetBProbabilityBJetTags = ak7PFbTagger.PositiveOnlyJetBProbabilityBJetTags

ak7PFSecondaryVertexTagInfos = ak7PFbTagger.SecondaryVertexTagInfos
ak7PFSimpleSecondaryVertexHighEffBJetTags = ak7PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak7PFSimpleSecondaryVertexHighPurBJetTags = ak7PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak7PFCombinedSecondaryVertexBJetTags = ak7PFbTagger.CombinedSecondaryVertexBJetTags
ak7PFCombinedSecondaryVertexV2BJetTags = ak7PFbTagger.CombinedSecondaryVertexV2BJetTags

ak7PFSecondaryVertexNegativeTagInfos = ak7PFbTagger.SecondaryVertexNegativeTagInfos
ak7PFNegativeSimpleSecondaryVertexHighEffBJetTags = ak7PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
ak7PFNegativeSimpleSecondaryVertexHighPurBJetTags = ak7PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
ak7PFNegativeCombinedSecondaryVertexBJetTags = ak7PFbTagger.NegativeCombinedSecondaryVertexBJetTags
ak7PFPositiveCombinedSecondaryVertexBJetTags = ak7PFbTagger.PositiveCombinedSecondaryVertexBJetTags

ak7PFSoftPFMuonsTagInfos = ak7PFbTagger.SoftPFMuonsTagInfos
ak7PFSoftPFMuonBJetTags = ak7PFbTagger.SoftPFMuonBJetTags
ak7PFSoftPFMuonByIP3dBJetTags = ak7PFbTagger.SoftPFMuonByIP3dBJetTags
ak7PFSoftPFMuonByPtBJetTags = ak7PFbTagger.SoftPFMuonByPtBJetTags
ak7PFNegativeSoftPFMuonByPtBJetTags = ak7PFbTagger.NegativeSoftPFMuonByPtBJetTags
ak7PFPositiveSoftPFMuonByPtBJetTags = ak7PFbTagger.PositiveSoftPFMuonByPtBJetTags
ak7PFPatJetFlavourIdLegacy = cms.Sequence(ak7PFPatJetPartonAssociationLegacy*ak7PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#ak7PFPatJetFlavourAssociation = ak7PFbTagger.PatJetFlavourAssociation
#ak7PFPatJetFlavourId = cms.Sequence(ak7PFPatJetPartons*ak7PFPatJetFlavourAssociation)

ak7PFJetBtaggingIP       = cms.Sequence(ak7PFImpactParameterTagInfos *
            (ak7PFTrackCountingHighEffBJetTags +
             ak7PFTrackCountingHighPurBJetTags +
             ak7PFJetProbabilityBJetTags +
             ak7PFJetBProbabilityBJetTags +
             ak7PFPositiveOnlyJetProbabilityBJetTags +
             ak7PFNegativeOnlyJetProbabilityBJetTags +
             ak7PFNegativeTrackCountingHighEffBJetTags +
             ak7PFNegativeTrackCountingHighPurBJetTags +
             ak7PFNegativeOnlyJetBProbabilityBJetTags +
             ak7PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

ak7PFJetBtaggingSV = cms.Sequence(ak7PFImpactParameterTagInfos
            *
            ak7PFSecondaryVertexTagInfos
            * (ak7PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak7PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak7PFCombinedSecondaryVertexBJetTags
                +
                ak7PFCombinedSecondaryVertexV2BJetTags
              )
            )

ak7PFJetBtaggingNegSV = cms.Sequence(ak7PFImpactParameterTagInfos
            *
            ak7PFSecondaryVertexNegativeTagInfos
            * (ak7PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                ak7PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                ak7PFNegativeCombinedSecondaryVertexBJetTags
                +
                ak7PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

ak7PFJetBtaggingMu = cms.Sequence(ak7PFSoftPFMuonsTagInfos * (ak7PFSoftPFMuonBJetTags
                +
                ak7PFSoftPFMuonByIP3dBJetTags
                +
                ak7PFSoftPFMuonByPtBJetTags
                +
                ak7PFNegativeSoftPFMuonByPtBJetTags
                +
                ak7PFPositiveSoftPFMuonByPtBJetTags
              )
            )

ak7PFJetBtagging = cms.Sequence(ak7PFJetBtaggingIP
            *ak7PFJetBtaggingSV
            *ak7PFJetBtaggingNegSV
            *ak7PFJetBtaggingMu
            )

ak7PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak7PFJets"),
        genJetMatch          = cms.InputTag("ak7PFmatch"),
        genPartonMatch       = cms.InputTag("ak7PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak7PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak7PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("ak7PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak7PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("ak7PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak7PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak7PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak7PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("ak7PFJetBProbabilityBJetTags"),
            cms.InputTag("ak7PFJetProbabilityBJetTags"),
            cms.InputTag("ak7PFSoftPFMuonByPtBJetTags"),
            cms.InputTag("ak7PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("ak7PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak7PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak7PFJetID"),
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

ak7PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak7PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak7HiGenJets',
                                                             rParam = 0.7,
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
                                                             bTagJetName = cms.untracked.string("ak7PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL')
                                                             )

ak7PFJetSequence_mc = cms.Sequence(
                                                  #ak7PFclean
                                                  #*
                                                  ak7PFmatch
                                                  *
                                                  ak7PFparton
                                                  *
                                                  ak7PFcorr
                                                  *
                                                  ak7PFJetID
                                                  *
                                                  ak7PFPatJetFlavourIdLegacy
                                                  #*
			                          #ak7PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  ak7PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak7PFJetBtagging
                                                  *
                                                  ak7PFpatJetsWithBtagging
                                                  *
                                                  ak7PFJetAnalyzer
                                                  )

ak7PFJetSequence_data = cms.Sequence(ak7PFcorr
                                                    *
                                                    ak7PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak7PFJetBtagging
                                                    *
                                                    ak7PFpatJetsWithBtagging
                                                    *
                                                    ak7PFJetAnalyzer
                                                    )

ak7PFJetSequence_jec = cms.Sequence(ak7PFJetSequence_mc)
ak7PFJetSequence_mix = cms.Sequence(ak7PFJetSequence_mc)

ak7PFJetSequence = cms.Sequence(ak7PFJetSequence_mix)
