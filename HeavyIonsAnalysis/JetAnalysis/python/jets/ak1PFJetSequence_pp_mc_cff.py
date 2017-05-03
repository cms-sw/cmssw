

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

ak1PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak1PFJets"),
    matched = cms.InputTag("ak1GenJets"),
    maxDeltaR = 0.1
    )

ak1PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak1PFJets")
                                                        )

ak1PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak1PFJets"),
    payload = "AK1PF_offline"
    )

ak1PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak1CaloJets'))

#ak1PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1GenJets'))

ak1PFbTagger = bTaggers("ak1PF",0.1)

#create objects locally since they dont load properly otherwise
#ak1PFmatch = ak1PFbTagger.match
ak1PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak1PFJets"), matched = cms.InputTag("genParticles"))
ak1PFPatJetFlavourAssociationLegacy = ak1PFbTagger.PatJetFlavourAssociationLegacy
ak1PFPatJetPartons = ak1PFbTagger.PatJetPartons
ak1PFJetTracksAssociatorAtVertex = ak1PFbTagger.JetTracksAssociatorAtVertex
ak1PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
ak1PFSimpleSecondaryVertexHighEffBJetTags = ak1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak1PFSimpleSecondaryVertexHighPurBJetTags = ak1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak1PFCombinedSecondaryVertexBJetTags = ak1PFbTagger.CombinedSecondaryVertexBJetTags
ak1PFCombinedSecondaryVertexV2BJetTags = ak1PFbTagger.CombinedSecondaryVertexV2BJetTags
ak1PFJetBProbabilityBJetTags = ak1PFbTagger.JetBProbabilityBJetTags
ak1PFSoftPFMuonByPtBJetTags = ak1PFbTagger.SoftPFMuonByPtBJetTags
ak1PFSoftPFMuonByIP3dBJetTags = ak1PFbTagger.SoftPFMuonByIP3dBJetTags
ak1PFTrackCountingHighEffBJetTags = ak1PFbTagger.TrackCountingHighEffBJetTags
ak1PFTrackCountingHighPurBJetTags = ak1PFbTagger.TrackCountingHighPurBJetTags
ak1PFPatJetPartonAssociationLegacy = ak1PFbTagger.PatJetPartonAssociationLegacy

ak1PFImpactParameterTagInfos = ak1PFbTagger.ImpactParameterTagInfos
ak1PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak1PFJetProbabilityBJetTags = ak1PFbTagger.JetProbabilityBJetTags
ak1PFPositiveOnlyJetProbabilityBJetTags = ak1PFbTagger.PositiveOnlyJetProbabilityBJetTags
ak1PFNegativeOnlyJetProbabilityBJetTags = ak1PFbTagger.NegativeOnlyJetProbabilityBJetTags
ak1PFNegativeTrackCountingHighEffBJetTags = ak1PFbTagger.NegativeTrackCountingHighEffBJetTags
ak1PFNegativeTrackCountingHighPurBJetTags = ak1PFbTagger.NegativeTrackCountingHighPurBJetTags
ak1PFNegativeOnlyJetBProbabilityBJetTags = ak1PFbTagger.NegativeOnlyJetBProbabilityBJetTags
ak1PFPositiveOnlyJetBProbabilityBJetTags = ak1PFbTagger.PositiveOnlyJetBProbabilityBJetTags

ak1PFSecondaryVertexTagInfos = ak1PFbTagger.SecondaryVertexTagInfos
ak1PFSimpleSecondaryVertexHighEffBJetTags = ak1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak1PFSimpleSecondaryVertexHighPurBJetTags = ak1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak1PFCombinedSecondaryVertexBJetTags = ak1PFbTagger.CombinedSecondaryVertexBJetTags
ak1PFCombinedSecondaryVertexV2BJetTags = ak1PFbTagger.CombinedSecondaryVertexV2BJetTags

ak1PFSecondaryVertexNegativeTagInfos = ak1PFbTagger.SecondaryVertexNegativeTagInfos
ak1PFNegativeSimpleSecondaryVertexHighEffBJetTags = ak1PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
ak1PFNegativeSimpleSecondaryVertexHighPurBJetTags = ak1PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
ak1PFNegativeCombinedSecondaryVertexBJetTags = ak1PFbTagger.NegativeCombinedSecondaryVertexBJetTags
ak1PFPositiveCombinedSecondaryVertexBJetTags = ak1PFbTagger.PositiveCombinedSecondaryVertexBJetTags

ak1PFSoftPFMuonsTagInfos = ak1PFbTagger.SoftPFMuonsTagInfos
ak1PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak1PFSoftPFMuonBJetTags = ak1PFbTagger.SoftPFMuonBJetTags
ak1PFSoftPFMuonByIP3dBJetTags = ak1PFbTagger.SoftPFMuonByIP3dBJetTags
ak1PFSoftPFMuonByPtBJetTags = ak1PFbTagger.SoftPFMuonByPtBJetTags
ak1PFNegativeSoftPFMuonByPtBJetTags = ak1PFbTagger.NegativeSoftPFMuonByPtBJetTags
ak1PFPositiveSoftPFMuonByPtBJetTags = ak1PFbTagger.PositiveSoftPFMuonByPtBJetTags
ak1PFPatJetFlavourIdLegacy = cms.Sequence(ak1PFPatJetPartonAssociationLegacy*ak1PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#ak1PFPatJetFlavourAssociation = ak1PFbTagger.PatJetFlavourAssociation
#ak1PFPatJetFlavourId = cms.Sequence(ak1PFPatJetPartons*ak1PFPatJetFlavourAssociation)

ak1PFJetBtaggingIP       = cms.Sequence(ak1PFImpactParameterTagInfos *
            (ak1PFTrackCountingHighEffBJetTags +
             ak1PFTrackCountingHighPurBJetTags +
             ak1PFJetProbabilityBJetTags +
             ak1PFJetBProbabilityBJetTags +
             ak1PFPositiveOnlyJetProbabilityBJetTags +
             ak1PFNegativeOnlyJetProbabilityBJetTags +
             ak1PFNegativeTrackCountingHighEffBJetTags +
             ak1PFNegativeTrackCountingHighPurBJetTags +
             ak1PFNegativeOnlyJetBProbabilityBJetTags +
             ak1PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

ak1PFJetBtaggingSV = cms.Sequence(ak1PFImpactParameterTagInfos
            *
            ak1PFSecondaryVertexTagInfos
            * (ak1PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak1PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak1PFCombinedSecondaryVertexBJetTags
                +
                ak1PFCombinedSecondaryVertexV2BJetTags
              )
            )

ak1PFJetBtaggingNegSV = cms.Sequence(ak1PFImpactParameterTagInfos
            *
            ak1PFSecondaryVertexNegativeTagInfos
            * (ak1PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                ak1PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                ak1PFNegativeCombinedSecondaryVertexBJetTags
                +
                ak1PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

ak1PFJetBtaggingMu = cms.Sequence(ak1PFSoftPFMuonsTagInfos * (ak1PFSoftPFMuonBJetTags
                +
                ak1PFSoftPFMuonByIP3dBJetTags
                +
                ak1PFSoftPFMuonByPtBJetTags
                +
                ak1PFNegativeSoftPFMuonByPtBJetTags
                +
                ak1PFPositiveSoftPFMuonByPtBJetTags
              )
            )

ak1PFJetBtagging = cms.Sequence(ak1PFJetBtaggingIP
            *ak1PFJetBtaggingSV
            *ak1PFJetBtaggingNegSV
#            *ak1PFJetBtaggingMu
            )

ak1PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak1PFJets"),
        genJetMatch          = cms.InputTag("ak1PFmatch"),
        genPartonMatch       = cms.InputTag("ak1PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak1PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak1PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("ak1PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak1PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("ak1PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak1PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak1PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak1PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("ak1PFJetBProbabilityBJetTags"),
            cms.InputTag("ak1PFJetProbabilityBJetTags"),
            #cms.InputTag("ak1PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("ak1PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("ak1PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak1PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak1PFJetID"),
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

ak1PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak1PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak1GenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlow'),
                                                             trackTag = cms.InputTag("generalTracks"),
                                                             fillGenJets = True,
                                                             isMC = True,
							     doSubEvent = True,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("ak1PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False)
                                                             )

ak1PFJetSequence_mc = cms.Sequence(
                                                  #ak1PFclean
                                                  #*
                                                  ak1PFmatch
                                                  *
                                                  ak1PFparton
                                                  *
                                                  ak1PFcorr
                                                  *
                                                  #ak1PFJetID
                                                  #*
                                                  ak1PFPatJetFlavourIdLegacy
                                                  #*
			                          #ak1PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  ak1PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak1PFJetBtagging
                                                  *
                                                  ak1PFpatJetsWithBtagging
                                                  *
                                                  ak1PFJetAnalyzer
                                                  )

ak1PFJetSequence_data = cms.Sequence(ak1PFcorr
                                                    *
                                                    #ak1PFJetID
                                                    #*
                                                    ak1PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak1PFJetBtagging
                                                    *
                                                    ak1PFpatJetsWithBtagging
                                                    *
                                                    ak1PFJetAnalyzer
                                                    )

ak1PFJetSequence_jec = cms.Sequence(ak1PFJetSequence_mc)
ak1PFJetSequence_mix = cms.Sequence(ak1PFJetSequence_mc)

ak1PFJetSequence = cms.Sequence(ak1PFJetSequence_mc)
