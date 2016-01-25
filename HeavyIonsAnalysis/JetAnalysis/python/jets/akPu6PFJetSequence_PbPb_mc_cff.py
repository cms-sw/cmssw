

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *

akPu6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu6PFJets"),
    matched = cms.InputTag("ak6HiGenJets"),
    maxDeltaR = 0.6
    )

akPu6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu6PFJets")
                                                        )

akPu6PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu6PFJets"),
    payload = "AKPu6PF_offline"
    )

akPu6PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu6CaloJets'))

#akPu6PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiGenJets'))

akPu6PFbTagger = bTaggers("akPu6PF",0.6)

#create objects locally since they dont load properly otherwise
#akPu6PFmatch = akPu6PFbTagger.match
akPu6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu6PFJets"), matched = cms.InputTag("genParticles"))
akPu6PFPatJetFlavourAssociationLegacy = akPu6PFbTagger.PatJetFlavourAssociationLegacy
akPu6PFPatJetPartons = akPu6PFbTagger.PatJetPartons
akPu6PFJetTracksAssociatorAtVertex = akPu6PFbTagger.JetTracksAssociatorAtVertex
akPu6PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akPu6PFSimpleSecondaryVertexHighEffBJetTags = akPu6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu6PFSimpleSecondaryVertexHighPurBJetTags = akPu6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu6PFCombinedSecondaryVertexBJetTags = akPu6PFbTagger.CombinedSecondaryVertexBJetTags
akPu6PFCombinedSecondaryVertexV2BJetTags = akPu6PFbTagger.CombinedSecondaryVertexV2BJetTags
akPu6PFJetBProbabilityBJetTags = akPu6PFbTagger.JetBProbabilityBJetTags
akPu6PFSoftPFMuonByPtBJetTags = akPu6PFbTagger.SoftPFMuonByPtBJetTags
akPu6PFSoftPFMuonByIP3dBJetTags = akPu6PFbTagger.SoftPFMuonByIP3dBJetTags
akPu6PFTrackCountingHighEffBJetTags = akPu6PFbTagger.TrackCountingHighEffBJetTags
akPu6PFTrackCountingHighPurBJetTags = akPu6PFbTagger.TrackCountingHighPurBJetTags
akPu6PFPatJetPartonAssociationLegacy = akPu6PFbTagger.PatJetPartonAssociationLegacy

akPu6PFImpactParameterTagInfos = akPu6PFbTagger.ImpactParameterTagInfos
akPu6PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu6PFJetProbabilityBJetTags = akPu6PFbTagger.JetProbabilityBJetTags
akPu6PFPositiveOnlyJetProbabilityBJetTags = akPu6PFbTagger.PositiveOnlyJetProbabilityBJetTags
akPu6PFNegativeOnlyJetProbabilityBJetTags = akPu6PFbTagger.NegativeOnlyJetProbabilityBJetTags
akPu6PFNegativeTrackCountingHighEffBJetTags = akPu6PFbTagger.NegativeTrackCountingHighEffBJetTags
akPu6PFNegativeTrackCountingHighPurBJetTags = akPu6PFbTagger.NegativeTrackCountingHighPurBJetTags
akPu6PFNegativeOnlyJetBProbabilityBJetTags = akPu6PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akPu6PFPositiveOnlyJetBProbabilityBJetTags = akPu6PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akPu6PFSecondaryVertexTagInfos = akPu6PFbTagger.SecondaryVertexTagInfos
akPu6PFSimpleSecondaryVertexHighEffBJetTags = akPu6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu6PFSimpleSecondaryVertexHighPurBJetTags = akPu6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu6PFCombinedSecondaryVertexBJetTags = akPu6PFbTagger.CombinedSecondaryVertexBJetTags
akPu6PFCombinedSecondaryVertexV2BJetTags = akPu6PFbTagger.CombinedSecondaryVertexV2BJetTags

akPu6PFSecondaryVertexNegativeTagInfos = akPu6PFbTagger.SecondaryVertexNegativeTagInfos
akPu6PFNegativeSimpleSecondaryVertexHighEffBJetTags = akPu6PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akPu6PFNegativeSimpleSecondaryVertexHighPurBJetTags = akPu6PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akPu6PFNegativeCombinedSecondaryVertexBJetTags = akPu6PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akPu6PFPositiveCombinedSecondaryVertexBJetTags = akPu6PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akPu6PFSoftPFMuonsTagInfos = akPu6PFbTagger.SoftPFMuonsTagInfos
akPu6PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu6PFSoftPFMuonBJetTags = akPu6PFbTagger.SoftPFMuonBJetTags
akPu6PFSoftPFMuonByIP3dBJetTags = akPu6PFbTagger.SoftPFMuonByIP3dBJetTags
akPu6PFSoftPFMuonByPtBJetTags = akPu6PFbTagger.SoftPFMuonByPtBJetTags
akPu6PFNegativeSoftPFMuonByPtBJetTags = akPu6PFbTagger.NegativeSoftPFMuonByPtBJetTags
akPu6PFPositiveSoftPFMuonByPtBJetTags = akPu6PFbTagger.PositiveSoftPFMuonByPtBJetTags
akPu6PFPatJetFlavourIdLegacy = cms.Sequence(akPu6PFPatJetPartonAssociationLegacy*akPu6PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akPu6PFPatJetFlavourAssociation = akPu6PFbTagger.PatJetFlavourAssociation
#akPu6PFPatJetFlavourId = cms.Sequence(akPu6PFPatJetPartons*akPu6PFPatJetFlavourAssociation)

akPu6PFJetBtaggingIP       = cms.Sequence(akPu6PFImpactParameterTagInfos *
            (akPu6PFTrackCountingHighEffBJetTags +
             akPu6PFTrackCountingHighPurBJetTags +
             akPu6PFJetProbabilityBJetTags +
             akPu6PFJetBProbabilityBJetTags +
             akPu6PFPositiveOnlyJetProbabilityBJetTags +
             akPu6PFNegativeOnlyJetProbabilityBJetTags +
             akPu6PFNegativeTrackCountingHighEffBJetTags +
             akPu6PFNegativeTrackCountingHighPurBJetTags +
             akPu6PFNegativeOnlyJetBProbabilityBJetTags +
             akPu6PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akPu6PFJetBtaggingSV = cms.Sequence(akPu6PFImpactParameterTagInfos
            *
            akPu6PFSecondaryVertexTagInfos
            * (akPu6PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu6PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu6PFCombinedSecondaryVertexBJetTags
                +
                akPu6PFCombinedSecondaryVertexV2BJetTags
              )
            )

akPu6PFJetBtaggingNegSV = cms.Sequence(akPu6PFImpactParameterTagInfos
            *
            akPu6PFSecondaryVertexNegativeTagInfos
            * (akPu6PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akPu6PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akPu6PFNegativeCombinedSecondaryVertexBJetTags
                +
                akPu6PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akPu6PFJetBtaggingMu = cms.Sequence(akPu6PFSoftPFMuonsTagInfos * (akPu6PFSoftPFMuonBJetTags
                +
                akPu6PFSoftPFMuonByIP3dBJetTags
                +
                akPu6PFSoftPFMuonByPtBJetTags
                +
                akPu6PFNegativeSoftPFMuonByPtBJetTags
                +
                akPu6PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akPu6PFJetBtagging = cms.Sequence(akPu6PFJetBtaggingIP
            *akPu6PFJetBtaggingSV
            *akPu6PFJetBtaggingNegSV
#            *akPu6PFJetBtaggingMu
            )

akPu6PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu6PFJets"),
        genJetMatch          = cms.InputTag("akPu6PFmatch"),
        genPartonMatch       = cms.InputTag("akPu6PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu6PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu6PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akPu6PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu6PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu6PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu6PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu6PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu6PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akPu6PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu6PFJetProbabilityBJetTags"),
            #cms.InputTag("akPu6PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akPu6PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akPu6PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu6PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu6PFJetID"),
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

akPu6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu6PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akPu6PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akPu6PFJetSequence_mc = cms.Sequence(
                                                  #akPu6PFclean
                                                  #*
                                                  akPu6PFmatch
                                                  *
                                                  akPu6PFparton
                                                  *
                                                  akPu6PFcorr
                                                  *
                                                  #akPu6PFJetID
                                                  #*
                                                  akPu6PFPatJetFlavourIdLegacy
                                                  #*
			                          #akPu6PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akPu6PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu6PFJetBtagging
                                                  *
                                                  akPu6PFpatJetsWithBtagging
                                                  *
                                                  akPu6PFJetAnalyzer
                                                  )

akPu6PFJetSequence_data = cms.Sequence(akPu6PFcorr
                                                    *
                                                    #akPu6PFJetID
                                                    #*
                                                    akPu6PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu6PFJetBtagging
                                                    *
                                                    akPu6PFpatJetsWithBtagging
                                                    *
                                                    akPu6PFJetAnalyzer
                                                    )

akPu6PFJetSequence_jec = cms.Sequence(akPu6PFJetSequence_mc)
akPu6PFJetSequence_mix = cms.Sequence(akPu6PFJetSequence_mc)

akPu6PFJetSequence = cms.Sequence(akPu6PFJetSequence_mc)
