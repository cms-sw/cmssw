

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akVs2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs2PFJets"),
    matched = cms.InputTag("ak2HiGenJets"),
    maxDeltaR = 0.2
    )

akVs2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs2PFJets")
                                                        )

akVs2PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs2PFJets"),
    payload = "AK2PF_offline"
    )

akVs2PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs2CaloJets'))

#akVs2PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiGenJets'))

akVs2PFbTagger = bTaggers("akVs2PF",0.2)

#create objects locally since they dont load properly otherwise
#akVs2PFmatch = akVs2PFbTagger.match
akVs2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs2PFJets"), matched = cms.InputTag("genParticles"))
akVs2PFPatJetFlavourAssociationLegacy = akVs2PFbTagger.PatJetFlavourAssociationLegacy
akVs2PFPatJetPartons = akVs2PFbTagger.PatJetPartons
akVs2PFJetTracksAssociatorAtVertex = akVs2PFbTagger.JetTracksAssociatorAtVertex
akVs2PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs2PFSimpleSecondaryVertexHighEffBJetTags = akVs2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs2PFSimpleSecondaryVertexHighPurBJetTags = akVs2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs2PFCombinedSecondaryVertexBJetTags = akVs2PFbTagger.CombinedSecondaryVertexBJetTags
akVs2PFCombinedSecondaryVertexV2BJetTags = akVs2PFbTagger.CombinedSecondaryVertexV2BJetTags
akVs2PFJetBProbabilityBJetTags = akVs2PFbTagger.JetBProbabilityBJetTags
akVs2PFSoftPFMuonByPtBJetTags = akVs2PFbTagger.SoftPFMuonByPtBJetTags
akVs2PFSoftPFMuonByIP3dBJetTags = akVs2PFbTagger.SoftPFMuonByIP3dBJetTags
akVs2PFTrackCountingHighEffBJetTags = akVs2PFbTagger.TrackCountingHighEffBJetTags
akVs2PFTrackCountingHighPurBJetTags = akVs2PFbTagger.TrackCountingHighPurBJetTags
akVs2PFPatJetPartonAssociationLegacy = akVs2PFbTagger.PatJetPartonAssociationLegacy

akVs2PFImpactParameterTagInfos = akVs2PFbTagger.ImpactParameterTagInfos
akVs2PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs2PFJetProbabilityBJetTags = akVs2PFbTagger.JetProbabilityBJetTags
akVs2PFPositiveOnlyJetProbabilityBJetTags = akVs2PFbTagger.PositiveOnlyJetProbabilityBJetTags
akVs2PFNegativeOnlyJetProbabilityBJetTags = akVs2PFbTagger.NegativeOnlyJetProbabilityBJetTags
akVs2PFNegativeTrackCountingHighEffBJetTags = akVs2PFbTagger.NegativeTrackCountingHighEffBJetTags
akVs2PFNegativeTrackCountingHighPurBJetTags = akVs2PFbTagger.NegativeTrackCountingHighPurBJetTags
akVs2PFNegativeOnlyJetBProbabilityBJetTags = akVs2PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akVs2PFPositiveOnlyJetBProbabilityBJetTags = akVs2PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akVs2PFSecondaryVertexTagInfos = akVs2PFbTagger.SecondaryVertexTagInfos
akVs2PFSimpleSecondaryVertexHighEffBJetTags = akVs2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs2PFSimpleSecondaryVertexHighPurBJetTags = akVs2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs2PFCombinedSecondaryVertexBJetTags = akVs2PFbTagger.CombinedSecondaryVertexBJetTags
akVs2PFCombinedSecondaryVertexV2BJetTags = akVs2PFbTagger.CombinedSecondaryVertexV2BJetTags

akVs2PFSecondaryVertexNegativeTagInfos = akVs2PFbTagger.SecondaryVertexNegativeTagInfos
akVs2PFNegativeSimpleSecondaryVertexHighEffBJetTags = akVs2PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs2PFNegativeSimpleSecondaryVertexHighPurBJetTags = akVs2PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs2PFNegativeCombinedSecondaryVertexBJetTags = akVs2PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akVs2PFPositiveCombinedSecondaryVertexBJetTags = akVs2PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akVs2PFSoftPFMuonsTagInfos = akVs2PFbTagger.SoftPFMuonsTagInfos
akVs2PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs2PFSoftPFMuonBJetTags = akVs2PFbTagger.SoftPFMuonBJetTags
akVs2PFSoftPFMuonByIP3dBJetTags = akVs2PFbTagger.SoftPFMuonByIP3dBJetTags
akVs2PFSoftPFMuonByPtBJetTags = akVs2PFbTagger.SoftPFMuonByPtBJetTags
akVs2PFNegativeSoftPFMuonByPtBJetTags = akVs2PFbTagger.NegativeSoftPFMuonByPtBJetTags
akVs2PFPositiveSoftPFMuonByPtBJetTags = akVs2PFbTagger.PositiveSoftPFMuonByPtBJetTags
akVs2PFPatJetFlavourIdLegacy = cms.Sequence(akVs2PFPatJetPartonAssociationLegacy*akVs2PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs2PFPatJetFlavourAssociation = akVs2PFbTagger.PatJetFlavourAssociation
#akVs2PFPatJetFlavourId = cms.Sequence(akVs2PFPatJetPartons*akVs2PFPatJetFlavourAssociation)

akVs2PFJetBtaggingIP       = cms.Sequence(akVs2PFImpactParameterTagInfos *
            (akVs2PFTrackCountingHighEffBJetTags +
             akVs2PFTrackCountingHighPurBJetTags +
             akVs2PFJetProbabilityBJetTags +
             akVs2PFJetBProbabilityBJetTags +
             akVs2PFPositiveOnlyJetProbabilityBJetTags +
             akVs2PFNegativeOnlyJetProbabilityBJetTags +
             akVs2PFNegativeTrackCountingHighEffBJetTags +
             akVs2PFNegativeTrackCountingHighPurBJetTags +
             akVs2PFNegativeOnlyJetBProbabilityBJetTags +
             akVs2PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs2PFJetBtaggingSV = cms.Sequence(akVs2PFImpactParameterTagInfos
            *
            akVs2PFSecondaryVertexTagInfos
            * (akVs2PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs2PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs2PFCombinedSecondaryVertexBJetTags
                +
                akVs2PFCombinedSecondaryVertexV2BJetTags
              )
            )

akVs2PFJetBtaggingNegSV = cms.Sequence(akVs2PFImpactParameterTagInfos
            *
            akVs2PFSecondaryVertexNegativeTagInfos
            * (akVs2PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs2PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs2PFNegativeCombinedSecondaryVertexBJetTags
                +
                akVs2PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs2PFJetBtaggingMu = cms.Sequence(akVs2PFSoftPFMuonsTagInfos * (akVs2PFSoftPFMuonBJetTags
                +
                akVs2PFSoftPFMuonByIP3dBJetTags
                +
                akVs2PFSoftPFMuonByPtBJetTags
                +
                akVs2PFNegativeSoftPFMuonByPtBJetTags
                +
                akVs2PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs2PFJetBtagging = cms.Sequence(akVs2PFJetBtaggingIP
            *akVs2PFJetBtaggingSV
            *akVs2PFJetBtaggingNegSV
#            *akVs2PFJetBtaggingMu
            )

akVs2PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs2PFJets"),
        genJetMatch          = cms.InputTag("akVs2PFmatch"),
        genPartonMatch       = cms.InputTag("akVs2PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs2PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs2PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs2PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs2PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs2PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs2PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs2PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs2PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs2PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs2PFJetProbabilityBJetTags"),
            #cms.InputTag("akVs2PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs2PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs2PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs2PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs2PFJetID"),
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

akVs2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs2PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akVs2PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akVs2PFJetSequence_mc = cms.Sequence(
                                                  #akVs2PFclean
                                                  #*
                                                  akVs2PFmatch
                                                  *
                                                  akVs2PFparton
                                                  *
                                                  akVs2PFcorr
                                                  *
                                                  #akVs2PFJetID
                                                  #*
                                                  akVs2PFPatJetFlavourIdLegacy
                                                  #*
			                          #akVs2PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs2PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs2PFJetBtagging
                                                  *
                                                  akVs2PFpatJetsWithBtagging
                                                  *
                                                  akVs2PFJetAnalyzer
                                                  )

akVs2PFJetSequence_data = cms.Sequence(akVs2PFcorr
                                                    *
                                                    #akVs2PFJetID
                                                    #*
                                                    akVs2PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs2PFJetBtagging
                                                    *
                                                    akVs2PFpatJetsWithBtagging
                                                    *
                                                    akVs2PFJetAnalyzer
                                                    )

akVs2PFJetSequence_jec = cms.Sequence(akVs2PFJetSequence_mc)
akVs2PFJetSequence_mix = cms.Sequence(akVs2PFJetSequence_mc)

akVs2PFJetSequence = cms.Sequence(akVs2PFJetSequence_jec)
akVs2PFJetAnalyzer.genPtMin = cms.untracked.double(1)
akVs2PFJetAnalyzer.jetPtMin = cms.untracked.double(1)
