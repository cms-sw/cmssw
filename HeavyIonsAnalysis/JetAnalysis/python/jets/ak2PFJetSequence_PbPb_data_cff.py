

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

ak2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("ak2PFJets"),
    matched = cms.InputTag("ak2HiGenJets"),
    maxDeltaR = 0.2
    )

ak2PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak2PFJets")
                                                        )

ak2PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("ak2PFJets"),
    payload = "AK2PF_offline"
    )

ak2PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('ak2CaloJets'))

#ak2PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJets'))

ak2PFbTagger = bTaggers("ak2PF",0.2)

#create objects locally since they dont load properly otherwise
#ak2PFmatch = ak2PFbTagger.match
ak2PFparton = patJetPartonMatch.clone(src = cms.InputTag("ak2PFJets"), matched = cms.InputTag("genParticles"))
ak2PFPatJetFlavourAssociationLegacy = ak2PFbTagger.PatJetFlavourAssociationLegacy
ak2PFPatJetPartons = ak2PFbTagger.PatJetPartons
ak2PFJetTracksAssociatorAtVertex = ak2PFbTagger.JetTracksAssociatorAtVertex
ak2PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
ak2PFSimpleSecondaryVertexHighEffBJetTags = ak2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak2PFSimpleSecondaryVertexHighPurBJetTags = ak2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak2PFCombinedSecondaryVertexBJetTags = ak2PFbTagger.CombinedSecondaryVertexBJetTags
ak2PFCombinedSecondaryVertexV2BJetTags = ak2PFbTagger.CombinedSecondaryVertexV2BJetTags
ak2PFJetBProbabilityBJetTags = ak2PFbTagger.JetBProbabilityBJetTags
ak2PFSoftPFMuonByPtBJetTags = ak2PFbTagger.SoftPFMuonByPtBJetTags
ak2PFSoftPFMuonByIP3dBJetTags = ak2PFbTagger.SoftPFMuonByIP3dBJetTags
ak2PFTrackCountingHighEffBJetTags = ak2PFbTagger.TrackCountingHighEffBJetTags
ak2PFTrackCountingHighPurBJetTags = ak2PFbTagger.TrackCountingHighPurBJetTags
ak2PFPatJetPartonAssociationLegacy = ak2PFbTagger.PatJetPartonAssociationLegacy

ak2PFImpactParameterTagInfos = ak2PFbTagger.ImpactParameterTagInfos
ak2PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak2PFJetProbabilityBJetTags = ak2PFbTagger.JetProbabilityBJetTags
ak2PFPositiveOnlyJetProbabilityBJetTags = ak2PFbTagger.PositiveOnlyJetProbabilityBJetTags
ak2PFNegativeOnlyJetProbabilityBJetTags = ak2PFbTagger.NegativeOnlyJetProbabilityBJetTags
ak2PFNegativeTrackCountingHighEffBJetTags = ak2PFbTagger.NegativeTrackCountingHighEffBJetTags
ak2PFNegativeTrackCountingHighPurBJetTags = ak2PFbTagger.NegativeTrackCountingHighPurBJetTags
ak2PFNegativeOnlyJetBProbabilityBJetTags = ak2PFbTagger.NegativeOnlyJetBProbabilityBJetTags
ak2PFPositiveOnlyJetBProbabilityBJetTags = ak2PFbTagger.PositiveOnlyJetBProbabilityBJetTags

ak2PFSecondaryVertexTagInfos = ak2PFbTagger.SecondaryVertexTagInfos
ak2PFSimpleSecondaryVertexHighEffBJetTags = ak2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
ak2PFSimpleSecondaryVertexHighPurBJetTags = ak2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
ak2PFCombinedSecondaryVertexBJetTags = ak2PFbTagger.CombinedSecondaryVertexBJetTags
ak2PFCombinedSecondaryVertexV2BJetTags = ak2PFbTagger.CombinedSecondaryVertexV2BJetTags

ak2PFSecondaryVertexNegativeTagInfos = ak2PFbTagger.SecondaryVertexNegativeTagInfos
ak2PFNegativeSimpleSecondaryVertexHighEffBJetTags = ak2PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
ak2PFNegativeSimpleSecondaryVertexHighPurBJetTags = ak2PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
ak2PFNegativeCombinedSecondaryVertexBJetTags = ak2PFbTagger.NegativeCombinedSecondaryVertexBJetTags
ak2PFPositiveCombinedSecondaryVertexBJetTags = ak2PFbTagger.PositiveCombinedSecondaryVertexBJetTags

ak2PFSoftPFMuonsTagInfos = ak2PFbTagger.SoftPFMuonsTagInfos
ak2PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
ak2PFSoftPFMuonBJetTags = ak2PFbTagger.SoftPFMuonBJetTags
ak2PFSoftPFMuonByIP3dBJetTags = ak2PFbTagger.SoftPFMuonByIP3dBJetTags
ak2PFSoftPFMuonByPtBJetTags = ak2PFbTagger.SoftPFMuonByPtBJetTags
ak2PFNegativeSoftPFMuonByPtBJetTags = ak2PFbTagger.NegativeSoftPFMuonByPtBJetTags
ak2PFPositiveSoftPFMuonByPtBJetTags = ak2PFbTagger.PositiveSoftPFMuonByPtBJetTags
ak2PFPatJetFlavourIdLegacy = cms.Sequence(ak2PFPatJetPartonAssociationLegacy*ak2PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#ak2PFPatJetFlavourAssociation = ak2PFbTagger.PatJetFlavourAssociation
#ak2PFPatJetFlavourId = cms.Sequence(ak2PFPatJetPartons*ak2PFPatJetFlavourAssociation)

ak2PFJetBtaggingIP       = cms.Sequence(ak2PFImpactParameterTagInfos *
            (ak2PFTrackCountingHighEffBJetTags +
             ak2PFTrackCountingHighPurBJetTags +
             ak2PFJetProbabilityBJetTags +
             ak2PFJetBProbabilityBJetTags +
             ak2PFPositiveOnlyJetProbabilityBJetTags +
             ak2PFNegativeOnlyJetProbabilityBJetTags +
             ak2PFNegativeTrackCountingHighEffBJetTags +
             ak2PFNegativeTrackCountingHighPurBJetTags +
             ak2PFNegativeOnlyJetBProbabilityBJetTags +
             ak2PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

ak2PFJetBtaggingSV = cms.Sequence(ak2PFImpactParameterTagInfos
            *
            ak2PFSecondaryVertexTagInfos
            * (ak2PFSimpleSecondaryVertexHighEffBJetTags
                +
                ak2PFSimpleSecondaryVertexHighPurBJetTags
                +
                ak2PFCombinedSecondaryVertexBJetTags
                +
                ak2PFCombinedSecondaryVertexV2BJetTags
              )
            )

ak2PFJetBtaggingNegSV = cms.Sequence(ak2PFImpactParameterTagInfos
            *
            ak2PFSecondaryVertexNegativeTagInfos
            * (ak2PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                ak2PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                ak2PFNegativeCombinedSecondaryVertexBJetTags
                +
                ak2PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

ak2PFJetBtaggingMu = cms.Sequence(ak2PFSoftPFMuonsTagInfos * (ak2PFSoftPFMuonBJetTags
                +
                ak2PFSoftPFMuonByIP3dBJetTags
                +
                ak2PFSoftPFMuonByPtBJetTags
                +
                ak2PFNegativeSoftPFMuonByPtBJetTags
                +
                ak2PFPositiveSoftPFMuonByPtBJetTags
              )
            )

ak2PFJetBtagging = cms.Sequence(ak2PFJetBtaggingIP
            *ak2PFJetBtaggingSV
            *ak2PFJetBtaggingNegSV
#            *ak2PFJetBtaggingMu
            )

ak2PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("ak2PFJets"),
        genJetMatch          = cms.InputTag("ak2PFmatch"),
        genPartonMatch       = cms.InputTag("ak2PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("ak2PFcorr")),
        JetPartonMapSource   = cms.InputTag("ak2PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("ak2PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("ak2PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("ak2PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("ak2PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("ak2PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("ak2PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("ak2PFJetBProbabilityBJetTags"),
            cms.InputTag("ak2PFJetProbabilityBJetTags"),
            #cms.InputTag("ak2PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("ak2PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("ak2PFTrackCountingHighEffBJetTags"),
            cms.InputTag("ak2PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("ak2PFJetID"),
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

ak2PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("ak2PFJets"),
           	    R0  = cms.double( 0.2)
)
ak2PFpatJetsWithBtagging.userData.userFloats.src += ['ak2PFNjettiness:tau1','ak2PFNjettiness:tau2','ak2PFNjettiness:tau3']

ak2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("ak2PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("ak2PF"),
                                                             jetName = cms.untracked.string("ak2PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

ak2PFJetSequence_mc = cms.Sequence(
                                                  #ak2PFclean
                                                  #*
                                                  ak2PFmatch
                                                  *
                                                  ak2PFparton
                                                  *
                                                  ak2PFcorr
                                                  *
                                                  #ak2PFJetID
                                                  #*
                                                  ak2PFPatJetFlavourIdLegacy
                                                  #*
			                          #ak2PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  ak2PFJetTracksAssociatorAtVertex
                                                  *
                                                  ak2PFJetBtagging
                                                  *
                                                  ak2PFNjettiness
                                                  *
                                                  ak2PFpatJetsWithBtagging
                                                  *
                                                  ak2PFJetAnalyzer
                                                  )

ak2PFJetSequence_data = cms.Sequence(ak2PFcorr
                                                    *
                                                    #ak2PFJetID
                                                    #*
                                                    ak2PFJetTracksAssociatorAtVertex
                                                    *
                                                    ak2PFJetBtagging
                                                    *
                                                    ak2PFNjettiness 
                                                    *
                                                    ak2PFpatJetsWithBtagging
                                                    *
                                                    ak2PFJetAnalyzer
                                                    )

ak2PFJetSequence_jec = cms.Sequence(ak2PFJetSequence_mc)
ak2PFJetSequence_mix = cms.Sequence(ak2PFJetSequence_mc)

ak2PFJetSequence = cms.Sequence(ak2PFJetSequence_data)
