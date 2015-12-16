

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs3PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs3PFJets"),
    matched = cms.InputTag("ak3GenJets"),
    maxDeltaR = 0.3
    )

akVs3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs3PFJets")
                                                        )

akVs3PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs3PFJets"),
    payload = "AK3PF_offline"
    )

akVs3PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs3CaloJets'))

#akVs3PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3GenJets'))

akVs3PFbTagger = bTaggers("akVs3PF",0.3)

#create objects locally since they dont load properly otherwise
#akVs3PFmatch = akVs3PFbTagger.match
akVs3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs3PFJets"), matched = cms.InputTag("genParticles"))
akVs3PFPatJetFlavourAssociationLegacy = akVs3PFbTagger.PatJetFlavourAssociationLegacy
akVs3PFPatJetPartons = akVs3PFbTagger.PatJetPartons
akVs3PFJetTracksAssociatorAtVertex = akVs3PFbTagger.JetTracksAssociatorAtVertex
akVs3PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs3PFSimpleSecondaryVertexHighEffBJetTags = akVs3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs3PFSimpleSecondaryVertexHighPurBJetTags = akVs3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs3PFCombinedSecondaryVertexBJetTags = akVs3PFbTagger.CombinedSecondaryVertexBJetTags
akVs3PFCombinedSecondaryVertexV2BJetTags = akVs3PFbTagger.CombinedSecondaryVertexV2BJetTags
akVs3PFJetBProbabilityBJetTags = akVs3PFbTagger.JetBProbabilityBJetTags
akVs3PFSoftPFMuonByPtBJetTags = akVs3PFbTagger.SoftPFMuonByPtBJetTags
akVs3PFSoftPFMuonByIP3dBJetTags = akVs3PFbTagger.SoftPFMuonByIP3dBJetTags
akVs3PFTrackCountingHighEffBJetTags = akVs3PFbTagger.TrackCountingHighEffBJetTags
akVs3PFTrackCountingHighPurBJetTags = akVs3PFbTagger.TrackCountingHighPurBJetTags
akVs3PFPatJetPartonAssociationLegacy = akVs3PFbTagger.PatJetPartonAssociationLegacy

akVs3PFImpactParameterTagInfos = akVs3PFbTagger.ImpactParameterTagInfos
akVs3PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs3PFJetProbabilityBJetTags = akVs3PFbTagger.JetProbabilityBJetTags
akVs3PFPositiveOnlyJetProbabilityBJetTags = akVs3PFbTagger.PositiveOnlyJetProbabilityBJetTags
akVs3PFNegativeOnlyJetProbabilityBJetTags = akVs3PFbTagger.NegativeOnlyJetProbabilityBJetTags
akVs3PFNegativeTrackCountingHighEffBJetTags = akVs3PFbTagger.NegativeTrackCountingHighEffBJetTags
akVs3PFNegativeTrackCountingHighPurBJetTags = akVs3PFbTagger.NegativeTrackCountingHighPurBJetTags
akVs3PFNegativeOnlyJetBProbabilityBJetTags = akVs3PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akVs3PFPositiveOnlyJetBProbabilityBJetTags = akVs3PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akVs3PFSecondaryVertexTagInfos = akVs3PFbTagger.SecondaryVertexTagInfos
akVs3PFSimpleSecondaryVertexHighEffBJetTags = akVs3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs3PFSimpleSecondaryVertexHighPurBJetTags = akVs3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs3PFCombinedSecondaryVertexBJetTags = akVs3PFbTagger.CombinedSecondaryVertexBJetTags
akVs3PFCombinedSecondaryVertexV2BJetTags = akVs3PFbTagger.CombinedSecondaryVertexV2BJetTags

akVs3PFSecondaryVertexNegativeTagInfos = akVs3PFbTagger.SecondaryVertexNegativeTagInfos
akVs3PFNegativeSimpleSecondaryVertexHighEffBJetTags = akVs3PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs3PFNegativeSimpleSecondaryVertexHighPurBJetTags = akVs3PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs3PFNegativeCombinedSecondaryVertexBJetTags = akVs3PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akVs3PFPositiveCombinedSecondaryVertexBJetTags = akVs3PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akVs3PFSoftPFMuonsTagInfos = akVs3PFbTagger.SoftPFMuonsTagInfos
akVs3PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs3PFSoftPFMuonBJetTags = akVs3PFbTagger.SoftPFMuonBJetTags
akVs3PFSoftPFMuonByIP3dBJetTags = akVs3PFbTagger.SoftPFMuonByIP3dBJetTags
akVs3PFSoftPFMuonByPtBJetTags = akVs3PFbTagger.SoftPFMuonByPtBJetTags
akVs3PFNegativeSoftPFMuonByPtBJetTags = akVs3PFbTagger.NegativeSoftPFMuonByPtBJetTags
akVs3PFPositiveSoftPFMuonByPtBJetTags = akVs3PFbTagger.PositiveSoftPFMuonByPtBJetTags
akVs3PFPatJetFlavourIdLegacy = cms.Sequence(akVs3PFPatJetPartonAssociationLegacy*akVs3PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs3PFPatJetFlavourAssociation = akVs3PFbTagger.PatJetFlavourAssociation
#akVs3PFPatJetFlavourId = cms.Sequence(akVs3PFPatJetPartons*akVs3PFPatJetFlavourAssociation)

akVs3PFJetBtaggingIP       = cms.Sequence(akVs3PFImpactParameterTagInfos *
            (akVs3PFTrackCountingHighEffBJetTags +
             akVs3PFTrackCountingHighPurBJetTags +
             akVs3PFJetProbabilityBJetTags +
             akVs3PFJetBProbabilityBJetTags +
             akVs3PFPositiveOnlyJetProbabilityBJetTags +
             akVs3PFNegativeOnlyJetProbabilityBJetTags +
             akVs3PFNegativeTrackCountingHighEffBJetTags +
             akVs3PFNegativeTrackCountingHighPurBJetTags +
             akVs3PFNegativeOnlyJetBProbabilityBJetTags +
             akVs3PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs3PFJetBtaggingSV = cms.Sequence(akVs3PFImpactParameterTagInfos
            *
            akVs3PFSecondaryVertexTagInfos
            * (akVs3PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs3PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs3PFCombinedSecondaryVertexBJetTags
                +
                akVs3PFCombinedSecondaryVertexV2BJetTags
              )
            )

akVs3PFJetBtaggingNegSV = cms.Sequence(akVs3PFImpactParameterTagInfos
            *
            akVs3PFSecondaryVertexNegativeTagInfos
            * (akVs3PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs3PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs3PFNegativeCombinedSecondaryVertexBJetTags
                +
                akVs3PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs3PFJetBtaggingMu = cms.Sequence(akVs3PFSoftPFMuonsTagInfos * (akVs3PFSoftPFMuonBJetTags
                +
                akVs3PFSoftPFMuonByIP3dBJetTags
                +
                akVs3PFSoftPFMuonByPtBJetTags
                +
                akVs3PFNegativeSoftPFMuonByPtBJetTags
                +
                akVs3PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs3PFJetBtagging = cms.Sequence(akVs3PFJetBtaggingIP
            *akVs3PFJetBtaggingSV
            *akVs3PFJetBtaggingNegSV
#            *akVs3PFJetBtaggingMu
            )

akVs3PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs3PFJets"),
        genJetMatch          = cms.InputTag("akVs3PFmatch"),
        genPartonMatch       = cms.InputTag("akVs3PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs3PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs3PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs3PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs3PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs3PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs3PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs3PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs3PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs3PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs3PFJetProbabilityBJetTags"),
            #cms.InputTag("akVs3PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs3PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs3PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs3PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs3PFJetID"),
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

akVs3PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs3PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak3GenJets',
                                                             rParam = 0.3,
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
                                                             bTagJetName = cms.untracked.string("akVs3PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False)
                                                             )

akVs3PFJetSequence_mc = cms.Sequence(
                                                  #akVs3PFclean
                                                  #*
                                                  akVs3PFmatch
                                                  *
                                                  akVs3PFparton
                                                  *
                                                  akVs3PFcorr
                                                  *
                                                  #akVs3PFJetID
                                                  #*
                                                  akVs3PFPatJetFlavourIdLegacy
                                                  #*
			                          #akVs3PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs3PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs3PFJetBtagging
                                                  *
                                                  akVs3PFpatJetsWithBtagging
                                                  *
                                                  akVs3PFJetAnalyzer
                                                  )

akVs3PFJetSequence_data = cms.Sequence(akVs3PFcorr
                                                    *
                                                    #akVs3PFJetID
                                                    #*
                                                    akVs3PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs3PFJetBtagging
                                                    *
                                                    akVs3PFpatJetsWithBtagging
                                                    *
                                                    akVs3PFJetAnalyzer
                                                    )

akVs3PFJetSequence_jec = cms.Sequence(akVs3PFJetSequence_mc)
akVs3PFJetSequence_mix = cms.Sequence(akVs3PFJetSequence_mc)

akVs3PFJetSequence = cms.Sequence(akVs3PFJetSequence_mc)
