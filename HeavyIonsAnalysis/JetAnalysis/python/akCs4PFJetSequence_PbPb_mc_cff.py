import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs4PFJets"),
    matched = cms.InputTag("ak4HiGenJets"),
    maxDeltaR = 0.4
    )

akCs4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs4PFJets")
                                                        )

akCs4PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs4PFJets"),
    payload = "AK4PF_offline"
    )

akCs4PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs4CaloJets'))

#akCs4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJets'))

akCs4PFbTagger = bTaggers("akCs4PF",0.4)

#create objects locally since they dont load properly otherwise
#akCs4PFmatch = akCs4PFbTagger.match
akCs4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs4PFJets"), matched = cms.InputTag("genParticles"))
akCs4PFPatJetFlavourAssociationLegacy = akCs4PFbTagger.PatJetFlavourAssociationLegacy
akCs4PFPatJetPartons = akCs4PFbTagger.PatJetPartons
akCs4PFJetTracksAssociatorAtVertex = akCs4PFbTagger.JetTracksAssociatorAtVertex
akCs4PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs4PFSimpleSecondaryVertexHighEffBJetTags = akCs4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs4PFSimpleSecondaryVertexHighPurBJetTags = akCs4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs4PFCombinedSecondaryVertexBJetTags = akCs4PFbTagger.CombinedSecondaryVertexBJetTags
akCs4PFCombinedSecondaryVertexV2BJetTags = akCs4PFbTagger.CombinedSecondaryVertexV2BJetTags
akCs4PFJetBProbabilityBJetTags = akCs4PFbTagger.JetBProbabilityBJetTags
akCs4PFSoftPFMuonByPtBJetTags = akCs4PFbTagger.SoftPFMuonByPtBJetTags
akCs4PFSoftPFMuonByIP3dBJetTags = akCs4PFbTagger.SoftPFMuonByIP3dBJetTags
akCs4PFTrackCountingHighEffBJetTags = akCs4PFbTagger.TrackCountingHighEffBJetTags
akCs4PFTrackCountingHighPurBJetTags = akCs4PFbTagger.TrackCountingHighPurBJetTags
akCs4PFPatJetPartonAssociationLegacy = akCs4PFbTagger.PatJetPartonAssociationLegacy

akCs4PFImpactParameterTagInfos = akCs4PFbTagger.ImpactParameterTagInfos
akCs4PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs4PFJetProbabilityBJetTags = akCs4PFbTagger.JetProbabilityBJetTags
akCs4PFPositiveOnlyJetProbabilityBJetTags = akCs4PFbTagger.PositiveOnlyJetProbabilityBJetTags
akCs4PFNegativeOnlyJetProbabilityBJetTags = akCs4PFbTagger.NegativeOnlyJetProbabilityBJetTags
akCs4PFNegativeTrackCountingHighEffBJetTags = akCs4PFbTagger.NegativeTrackCountingHighEffBJetTags
akCs4PFNegativeTrackCountingHighPurBJetTags = akCs4PFbTagger.NegativeTrackCountingHighPurBJetTags
akCs4PFNegativeOnlyJetBProbabilityBJetTags = akCs4PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akCs4PFPositiveOnlyJetBProbabilityBJetTags = akCs4PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akCs4PFSecondaryVertexTagInfos = akCs4PFbTagger.SecondaryVertexTagInfos
akCs4PFSimpleSecondaryVertexHighEffBJetTags = akCs4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs4PFSimpleSecondaryVertexHighPurBJetTags = akCs4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs4PFCombinedSecondaryVertexBJetTags = akCs4PFbTagger.CombinedSecondaryVertexBJetTags
akCs4PFCombinedSecondaryVertexV2BJetTags = akCs4PFbTagger.CombinedSecondaryVertexV2BJetTags

akCs4PFSecondaryVertexNegativeTagInfos = akCs4PFbTagger.SecondaryVertexNegativeTagInfos
akCs4PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCs4PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs4PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCs4PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs4PFNegativeCombinedSecondaryVertexBJetTags = akCs4PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCs4PFPositiveCombinedSecondaryVertexBJetTags = akCs4PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akCs4PFSoftPFMuonsTagInfos = akCs4PFbTagger.SoftPFMuonsTagInfos
akCs4PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs4PFSoftPFMuonBJetTags = akCs4PFbTagger.SoftPFMuonBJetTags
akCs4PFSoftPFMuonByIP3dBJetTags = akCs4PFbTagger.SoftPFMuonByIP3dBJetTags
akCs4PFSoftPFMuonByPtBJetTags = akCs4PFbTagger.SoftPFMuonByPtBJetTags
akCs4PFNegativeSoftPFMuonByPtBJetTags = akCs4PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCs4PFPositiveSoftPFMuonByPtBJetTags = akCs4PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCs4PFPatJetFlavourIdLegacy = cms.Sequence(akCs4PFPatJetPartonAssociationLegacy*akCs4PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs4PFPatJetFlavourAssociation = akCs4PFbTagger.PatJetFlavourAssociation
#akCs4PFPatJetFlavourId = cms.Sequence(akCs4PFPatJetPartons*akCs4PFPatJetFlavourAssociation)

akCs4PFJetBtaggingIP       = cms.Sequence(akCs4PFImpactParameterTagInfos *
            (akCs4PFTrackCountingHighEffBJetTags +
             akCs4PFTrackCountingHighPurBJetTags +
             akCs4PFJetProbabilityBJetTags +
             akCs4PFJetBProbabilityBJetTags +
             akCs4PFPositiveOnlyJetProbabilityBJetTags +
             akCs4PFNegativeOnlyJetProbabilityBJetTags +
             akCs4PFNegativeTrackCountingHighEffBJetTags +
             akCs4PFNegativeTrackCountingHighPurBJetTags +
             akCs4PFNegativeOnlyJetBProbabilityBJetTags +
             akCs4PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCs4PFJetBtaggingSV = cms.Sequence(akCs4PFImpactParameterTagInfos
            *
            akCs4PFSecondaryVertexTagInfos
            * (akCs4PFSimpleSecondaryVertexHighEffBJetTags
                +
                akCs4PFSimpleSecondaryVertexHighPurBJetTags
                +
                akCs4PFCombinedSecondaryVertexBJetTags
                +
                akCs4PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCs4PFJetBtaggingNegSV = cms.Sequence(akCs4PFImpactParameterTagInfos
            *
            akCs4PFSecondaryVertexNegativeTagInfos
            * (akCs4PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCs4PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCs4PFNegativeCombinedSecondaryVertexBJetTags
                +
                akCs4PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCs4PFJetBtaggingMu = cms.Sequence(akCs4PFSoftPFMuonsTagInfos * (akCs4PFSoftPFMuonBJetTags
                +
                akCs4PFSoftPFMuonByIP3dBJetTags
                +
                akCs4PFSoftPFMuonByPtBJetTags
                +
                akCs4PFNegativeSoftPFMuonByPtBJetTags
                +
                akCs4PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs4PFJetBtagging = cms.Sequence(akCs4PFJetBtaggingIP
            *akCs4PFJetBtaggingSV
            *akCs4PFJetBtaggingNegSV
#            *akCs4PFJetBtaggingMu
            )

akCs4PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs4PFJets"),
        genJetMatch          = cms.InputTag("akCs4PFmatch"),
        genPartonMatch       = cms.InputTag("akCs4PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs4PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCs4PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs4PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs4PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs4PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs4PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs4PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs4PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs4PFJetBProbabilityBJetTags"),
            cms.InputTag("akCs4PFJetProbabilityBJetTags"),
            #cms.InputTag("akCs4PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs4PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs4PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs4PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs4PFJetID"),
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

akCs4PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs4PFJets"),
           	    R0  = cms.double( 0.4 )
)
akCs4PFpatJetsWithBtagging.userData.userFloats.src += ['akCs4PFNjettiness:tau1','akCs4PFNjettiness:tau2','akCs4PFNjettiness:tau3']

akCs4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs4PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak4HiGenJets',
                                                             rParam = 0.4,
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
                                                             bTagJetName = cms.untracked.string("akCs4PF"),
                                                             jetName = cms.untracked.string("akCs4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akCs4PFJetSequence_mc = cms.Sequence(
                                                  #akCs4PFclean
                                                  #*
                                                  akCs4PFmatch
                                                  *
                                                  akCs4PFparton
                                                  *
                                                  akCs4PFcorr
                                                  *
                                                  #akCs4PFJetID
                                                  #*
                                                  akCs4PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCs4PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs4PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCs4PFJetBtagging
                                                  *
						  akCs4PFNjettiness
                                                  *
                                                  akCs4PFpatJetsWithBtagging
						  *
						  akCs4PFJetAnalyzer
                                                  )

akCs4PFJetSequence_data = cms.Sequence(akCs4PFcorr
                                                    *
                                                    #akCs4PFJetID
                                                    #*
                                                    akCs4PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCs4PFJetBtagging
                                                    *
						    akCs4PFNjettiness
                                                    *
                                                    akCs4PFpatJetsWithBtagging
                                                    *
						    akCs4PFJetAnalyzer
                                                    )

akCs4PFJetSequence_jec = cms.Sequence(akCs4PFJetSequence_mc)
akCs4PFJetSequence_mix = cms.Sequence(akCs4PFJetSequence_mc)

akCs4PFJetSequence = cms.Sequence(akCs4PFJetSequence_mc)
