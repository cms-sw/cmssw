

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs6PFJets"),
    matched = cms.InputTag("ak6HiGenJets"),
    maxDeltaR = 0.6
    )

akVs6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs6PFJets")
                                                        )

akVs6PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs6PFJets"),
    payload = "AK6PF_offline"
    )

akVs6PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs6CaloJets'))

#akVs6PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiGenJets'))

akVs6PFbTagger = bTaggers("akVs6PF",0.6)

#create objects locally since they dont load properly otherwise
#akVs6PFmatch = akVs6PFbTagger.match
akVs6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs6PFJets"), matched = cms.InputTag("genParticles"))
akVs6PFPatJetFlavourAssociationLegacy = akVs6PFbTagger.PatJetFlavourAssociationLegacy
akVs6PFPatJetPartons = akVs6PFbTagger.PatJetPartons
akVs6PFJetTracksAssociatorAtVertex = akVs6PFbTagger.JetTracksAssociatorAtVertex
akVs6PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs6PFSimpleSecondaryVertexHighEffBJetTags = akVs6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs6PFSimpleSecondaryVertexHighPurBJetTags = akVs6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs6PFCombinedSecondaryVertexBJetTags = akVs6PFbTagger.CombinedSecondaryVertexBJetTags
akVs6PFCombinedSecondaryVertexV2BJetTags = akVs6PFbTagger.CombinedSecondaryVertexV2BJetTags
akVs6PFJetBProbabilityBJetTags = akVs6PFbTagger.JetBProbabilityBJetTags
akVs6PFSoftPFMuonByPtBJetTags = akVs6PFbTagger.SoftPFMuonByPtBJetTags
akVs6PFSoftPFMuonByIP3dBJetTags = akVs6PFbTagger.SoftPFMuonByIP3dBJetTags
akVs6PFTrackCountingHighEffBJetTags = akVs6PFbTagger.TrackCountingHighEffBJetTags
akVs6PFTrackCountingHighPurBJetTags = akVs6PFbTagger.TrackCountingHighPurBJetTags
akVs6PFPatJetPartonAssociationLegacy = akVs6PFbTagger.PatJetPartonAssociationLegacy

akVs6PFImpactParameterTagInfos = akVs6PFbTagger.ImpactParameterTagInfos
akVs6PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs6PFJetProbabilityBJetTags = akVs6PFbTagger.JetProbabilityBJetTags
akVs6PFPositiveOnlyJetProbabilityBJetTags = akVs6PFbTagger.PositiveOnlyJetProbabilityBJetTags
akVs6PFNegativeOnlyJetProbabilityBJetTags = akVs6PFbTagger.NegativeOnlyJetProbabilityBJetTags
akVs6PFNegativeTrackCountingHighEffBJetTags = akVs6PFbTagger.NegativeTrackCountingHighEffBJetTags
akVs6PFNegativeTrackCountingHighPurBJetTags = akVs6PFbTagger.NegativeTrackCountingHighPurBJetTags
akVs6PFNegativeOnlyJetBProbabilityBJetTags = akVs6PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akVs6PFPositiveOnlyJetBProbabilityBJetTags = akVs6PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akVs6PFSecondaryVertexTagInfos = akVs6PFbTagger.SecondaryVertexTagInfos
akVs6PFSimpleSecondaryVertexHighEffBJetTags = akVs6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs6PFSimpleSecondaryVertexHighPurBJetTags = akVs6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs6PFCombinedSecondaryVertexBJetTags = akVs6PFbTagger.CombinedSecondaryVertexBJetTags
akVs6PFCombinedSecondaryVertexV2BJetTags = akVs6PFbTagger.CombinedSecondaryVertexV2BJetTags

akVs6PFSecondaryVertexNegativeTagInfos = akVs6PFbTagger.SecondaryVertexNegativeTagInfos
akVs6PFNegativeSimpleSecondaryVertexHighEffBJetTags = akVs6PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs6PFNegativeSimpleSecondaryVertexHighPurBJetTags = akVs6PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs6PFNegativeCombinedSecondaryVertexBJetTags = akVs6PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akVs6PFPositiveCombinedSecondaryVertexBJetTags = akVs6PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akVs6PFSoftPFMuonsTagInfos = akVs6PFbTagger.SoftPFMuonsTagInfos
akVs6PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs6PFSoftPFMuonBJetTags = akVs6PFbTagger.SoftPFMuonBJetTags
akVs6PFSoftPFMuonByIP3dBJetTags = akVs6PFbTagger.SoftPFMuonByIP3dBJetTags
akVs6PFSoftPFMuonByPtBJetTags = akVs6PFbTagger.SoftPFMuonByPtBJetTags
akVs6PFNegativeSoftPFMuonByPtBJetTags = akVs6PFbTagger.NegativeSoftPFMuonByPtBJetTags
akVs6PFPositiveSoftPFMuonByPtBJetTags = akVs6PFbTagger.PositiveSoftPFMuonByPtBJetTags
akVs6PFPatJetFlavourIdLegacy = cms.Sequence(akVs6PFPatJetPartonAssociationLegacy*akVs6PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs6PFPatJetFlavourAssociation = akVs6PFbTagger.PatJetFlavourAssociation
#akVs6PFPatJetFlavourId = cms.Sequence(akVs6PFPatJetPartons*akVs6PFPatJetFlavourAssociation)

akVs6PFJetBtaggingIP       = cms.Sequence(akVs6PFImpactParameterTagInfos *
            (akVs6PFTrackCountingHighEffBJetTags +
             akVs6PFTrackCountingHighPurBJetTags +
             akVs6PFJetProbabilityBJetTags +
             akVs6PFJetBProbabilityBJetTags +
             akVs6PFPositiveOnlyJetProbabilityBJetTags +
             akVs6PFNegativeOnlyJetProbabilityBJetTags +
             akVs6PFNegativeTrackCountingHighEffBJetTags +
             akVs6PFNegativeTrackCountingHighPurBJetTags +
             akVs6PFNegativeOnlyJetBProbabilityBJetTags +
             akVs6PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs6PFJetBtaggingSV = cms.Sequence(akVs6PFImpactParameterTagInfos
            *
            akVs6PFSecondaryVertexTagInfos
            * (akVs6PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs6PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs6PFCombinedSecondaryVertexBJetTags
                +
                akVs6PFCombinedSecondaryVertexV2BJetTags
              )
            )

akVs6PFJetBtaggingNegSV = cms.Sequence(akVs6PFImpactParameterTagInfos
            *
            akVs6PFSecondaryVertexNegativeTagInfos
            * (akVs6PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs6PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs6PFNegativeCombinedSecondaryVertexBJetTags
                +
                akVs6PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs6PFJetBtaggingMu = cms.Sequence(akVs6PFSoftPFMuonsTagInfos * (akVs6PFSoftPFMuonBJetTags
                +
                akVs6PFSoftPFMuonByIP3dBJetTags
                +
                akVs6PFSoftPFMuonByPtBJetTags
                +
                akVs6PFNegativeSoftPFMuonByPtBJetTags
                +
                akVs6PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs6PFJetBtagging = cms.Sequence(akVs6PFJetBtaggingIP
            *akVs6PFJetBtaggingSV
            *akVs6PFJetBtaggingNegSV
#            *akVs6PFJetBtaggingMu
            )

akVs6PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs6PFJets"),
        genJetMatch          = cms.InputTag("akVs6PFmatch"),
        genPartonMatch       = cms.InputTag("akVs6PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs6PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs6PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs6PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs6PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs6PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs6PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs6PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs6PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs6PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs6PFJetProbabilityBJetTags"),
            #cms.InputTag("akVs6PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs6PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs6PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs6PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs6PFJetID"),
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

akVs6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs6PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
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
                                                             bTagJetName = cms.untracked.string("akVs6PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akVs6PFJetSequence_mc = cms.Sequence(
                                                  #akVs6PFclean
                                                  #*
                                                  akVs6PFmatch
                                                  *
                                                  akVs6PFparton
                                                  *
                                                  akVs6PFcorr
                                                  *
                                                  #akVs6PFJetID
                                                  #*
                                                  akVs6PFPatJetFlavourIdLegacy
                                                  #*
			                          #akVs6PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs6PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs6PFJetBtagging
                                                  *
                                                  akVs6PFpatJetsWithBtagging
                                                  *
                                                  akVs6PFJetAnalyzer
                                                  )

akVs6PFJetSequence_data = cms.Sequence(akVs6PFcorr
                                                    *
                                                    #akVs6PFJetID
                                                    #*
                                                    akVs6PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs6PFJetBtagging
                                                    *
                                                    akVs6PFpatJetsWithBtagging
                                                    *
                                                    akVs6PFJetAnalyzer
                                                    )

akVs6PFJetSequence_jec = cms.Sequence(akVs6PFJetSequence_mc)
akVs6PFJetSequence_mix = cms.Sequence(akVs6PFJetSequence_mc)

akVs6PFJetSequence = cms.Sequence(akVs6PFJetSequence_mc)
