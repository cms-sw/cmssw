

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs4PFJets"),
    matched = cms.InputTag("ak4GenJets"),
    maxDeltaR = 0.4
    )

akVs4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs4PFJets")
                                                        )

akVs4PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs4PFJets"),
    payload = "AK4PF_offline"
    )

akVs4PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs4CaloJets'))

#akVs4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4GenJets'))

akVs4PFbTagger = bTaggers("akVs4PF",0.4)

#create objects locally since they dont load properly otherwise
#akVs4PFmatch = akVs4PFbTagger.match
akVs4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs4PFJets"), matched = cms.InputTag("genParticles"))
akVs4PFPatJetFlavourAssociationLegacy = akVs4PFbTagger.PatJetFlavourAssociationLegacy
akVs4PFPatJetPartons = akVs4PFbTagger.PatJetPartons
akVs4PFJetTracksAssociatorAtVertex = akVs4PFbTagger.JetTracksAssociatorAtVertex
akVs4PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs4PFSimpleSecondaryVertexHighEffBJetTags = akVs4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs4PFSimpleSecondaryVertexHighPurBJetTags = akVs4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs4PFCombinedSecondaryVertexBJetTags = akVs4PFbTagger.CombinedSecondaryVertexBJetTags
akVs4PFCombinedSecondaryVertexV2BJetTags = akVs4PFbTagger.CombinedSecondaryVertexV2BJetTags
akVs4PFJetBProbabilityBJetTags = akVs4PFbTagger.JetBProbabilityBJetTags
akVs4PFSoftPFMuonByPtBJetTags = akVs4PFbTagger.SoftPFMuonByPtBJetTags
akVs4PFSoftPFMuonByIP3dBJetTags = akVs4PFbTagger.SoftPFMuonByIP3dBJetTags
akVs4PFTrackCountingHighEffBJetTags = akVs4PFbTagger.TrackCountingHighEffBJetTags
akVs4PFTrackCountingHighPurBJetTags = akVs4PFbTagger.TrackCountingHighPurBJetTags
akVs4PFPatJetPartonAssociationLegacy = akVs4PFbTagger.PatJetPartonAssociationLegacy

akVs4PFImpactParameterTagInfos = akVs4PFbTagger.ImpactParameterTagInfos
akVs4PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs4PFJetProbabilityBJetTags = akVs4PFbTagger.JetProbabilityBJetTags
akVs4PFPositiveOnlyJetProbabilityBJetTags = akVs4PFbTagger.PositiveOnlyJetProbabilityBJetTags
akVs4PFNegativeOnlyJetProbabilityBJetTags = akVs4PFbTagger.NegativeOnlyJetProbabilityBJetTags
akVs4PFNegativeTrackCountingHighEffBJetTags = akVs4PFbTagger.NegativeTrackCountingHighEffBJetTags
akVs4PFNegativeTrackCountingHighPurBJetTags = akVs4PFbTagger.NegativeTrackCountingHighPurBJetTags
akVs4PFNegativeOnlyJetBProbabilityBJetTags = akVs4PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akVs4PFPositiveOnlyJetBProbabilityBJetTags = akVs4PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akVs4PFSecondaryVertexTagInfos = akVs4PFbTagger.SecondaryVertexTagInfos
akVs4PFSimpleSecondaryVertexHighEffBJetTags = akVs4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs4PFSimpleSecondaryVertexHighPurBJetTags = akVs4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs4PFCombinedSecondaryVertexBJetTags = akVs4PFbTagger.CombinedSecondaryVertexBJetTags
akVs4PFCombinedSecondaryVertexV2BJetTags = akVs4PFbTagger.CombinedSecondaryVertexV2BJetTags

akVs4PFSecondaryVertexNegativeTagInfos = akVs4PFbTagger.SecondaryVertexNegativeTagInfos
akVs4PFNegativeSimpleSecondaryVertexHighEffBJetTags = akVs4PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs4PFNegativeSimpleSecondaryVertexHighPurBJetTags = akVs4PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs4PFNegativeCombinedSecondaryVertexBJetTags = akVs4PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akVs4PFPositiveCombinedSecondaryVertexBJetTags = akVs4PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akVs4PFSoftPFMuonsTagInfos = akVs4PFbTagger.SoftPFMuonsTagInfos
akVs4PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs4PFSoftPFMuonBJetTags = akVs4PFbTagger.SoftPFMuonBJetTags
akVs4PFSoftPFMuonByIP3dBJetTags = akVs4PFbTagger.SoftPFMuonByIP3dBJetTags
akVs4PFSoftPFMuonByPtBJetTags = akVs4PFbTagger.SoftPFMuonByPtBJetTags
akVs4PFNegativeSoftPFMuonByPtBJetTags = akVs4PFbTagger.NegativeSoftPFMuonByPtBJetTags
akVs4PFPositiveSoftPFMuonByPtBJetTags = akVs4PFbTagger.PositiveSoftPFMuonByPtBJetTags
akVs4PFPatJetFlavourIdLegacy = cms.Sequence(akVs4PFPatJetPartonAssociationLegacy*akVs4PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs4PFPatJetFlavourAssociation = akVs4PFbTagger.PatJetFlavourAssociation
#akVs4PFPatJetFlavourId = cms.Sequence(akVs4PFPatJetPartons*akVs4PFPatJetFlavourAssociation)

akVs4PFJetBtaggingIP       = cms.Sequence(akVs4PFImpactParameterTagInfos *
            (akVs4PFTrackCountingHighEffBJetTags +
             akVs4PFTrackCountingHighPurBJetTags +
             akVs4PFJetProbabilityBJetTags +
             akVs4PFJetBProbabilityBJetTags +
             akVs4PFPositiveOnlyJetProbabilityBJetTags +
             akVs4PFNegativeOnlyJetProbabilityBJetTags +
             akVs4PFNegativeTrackCountingHighEffBJetTags +
             akVs4PFNegativeTrackCountingHighPurBJetTags +
             akVs4PFNegativeOnlyJetBProbabilityBJetTags +
             akVs4PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs4PFJetBtaggingSV = cms.Sequence(akVs4PFImpactParameterTagInfos
            *
            akVs4PFSecondaryVertexTagInfos
            * (akVs4PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs4PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs4PFCombinedSecondaryVertexBJetTags
                +
                akVs4PFCombinedSecondaryVertexV2BJetTags
              )
            )

akVs4PFJetBtaggingNegSV = cms.Sequence(akVs4PFImpactParameterTagInfos
            *
            akVs4PFSecondaryVertexNegativeTagInfos
            * (akVs4PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs4PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs4PFNegativeCombinedSecondaryVertexBJetTags
                +
                akVs4PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs4PFJetBtaggingMu = cms.Sequence(akVs4PFSoftPFMuonsTagInfos * (akVs4PFSoftPFMuonBJetTags
                +
                akVs4PFSoftPFMuonByIP3dBJetTags
                +
                akVs4PFSoftPFMuonByPtBJetTags
                +
                akVs4PFNegativeSoftPFMuonByPtBJetTags
                +
                akVs4PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs4PFJetBtagging = cms.Sequence(akVs4PFJetBtaggingIP
            *akVs4PFJetBtaggingSV
            *akVs4PFJetBtaggingNegSV
#            *akVs4PFJetBtaggingMu
            )

akVs4PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs4PFJets"),
        genJetMatch          = cms.InputTag("akVs4PFmatch"),
        genPartonMatch       = cms.InputTag("akVs4PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs4PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs4PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs4PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs4PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs4PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs4PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs4PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs4PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs4PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs4PFJetProbabilityBJetTags"),
            #cms.InputTag("akVs4PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs4PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs4PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs4PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs4PFJetID"),
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

akVs4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs4PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak4GenJets',
                                                             rParam = 0.4,
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
                                                             bTagJetName = cms.untracked.string("akVs4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(False)
                                                             )

akVs4PFJetSequence_mc = cms.Sequence(
                                                  #akVs4PFclean
                                                  #*
                                                  akVs4PFmatch
                                                  *
                                                  akVs4PFparton
                                                  *
                                                  akVs4PFcorr
                                                  *
                                                  #akVs4PFJetID
                                                  #*
                                                  akVs4PFPatJetFlavourIdLegacy
                                                  #*
			                          #akVs4PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs4PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs4PFJetBtagging
                                                  *
                                                  akVs4PFpatJetsWithBtagging
                                                  *
                                                  akVs4PFJetAnalyzer
                                                  )

akVs4PFJetSequence_data = cms.Sequence(akVs4PFcorr
                                                    *
                                                    #akVs4PFJetID
                                                    #*
                                                    akVs4PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs4PFJetBtagging
                                                    *
                                                    akVs4PFpatJetsWithBtagging
                                                    *
                                                    akVs4PFJetAnalyzer
                                                    )

akVs4PFJetSequence_jec = cms.Sequence(akVs4PFJetSequence_mc)
akVs4PFJetSequence_mix = cms.Sequence(akVs4PFJetSequence_mc)

akVs4PFJetSequence = cms.Sequence(akVs4PFJetSequence_mc)
