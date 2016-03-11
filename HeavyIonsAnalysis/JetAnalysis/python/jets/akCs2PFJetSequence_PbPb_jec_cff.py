

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs2PFJets"),
    matched = cms.InputTag("ak2HiSignalGenJets"),
    maxDeltaR = 0.2
    )

akCs2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs2PFJets")
                                                        )

akCs2PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs2PFJets"),
    payload = "AK2PF_offline"
    )

akCs2PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs2CaloJets'))

#akCs2PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiSignalGenJets'))

akCs2PFbTagger = bTaggers("akCs2PF",0.2)

#create objects locally since they dont load properly otherwise
#akCs2PFmatch = akCs2PFbTagger.match
akCs2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs2PFJets"), matched = cms.InputTag("selectedPartons"))
akCs2PFPatJetFlavourAssociationLegacy = akCs2PFbTagger.PatJetFlavourAssociationLegacy
akCs2PFPatJetPartons = akCs2PFbTagger.PatJetPartons
akCs2PFJetTracksAssociatorAtVertex = akCs2PFbTagger.JetTracksAssociatorAtVertex
akCs2PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs2PFSimpleSecondaryVertexHighEffBJetTags = akCs2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs2PFSimpleSecondaryVertexHighPurBJetTags = akCs2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs2PFCombinedSecondaryVertexBJetTags = akCs2PFbTagger.CombinedSecondaryVertexBJetTags
akCs2PFCombinedSecondaryVertexV2BJetTags = akCs2PFbTagger.CombinedSecondaryVertexV2BJetTags
akCs2PFJetBProbabilityBJetTags = akCs2PFbTagger.JetBProbabilityBJetTags
akCs2PFSoftPFMuonByPtBJetTags = akCs2PFbTagger.SoftPFMuonByPtBJetTags
akCs2PFSoftPFMuonByIP3dBJetTags = akCs2PFbTagger.SoftPFMuonByIP3dBJetTags
akCs2PFTrackCountingHighEffBJetTags = akCs2PFbTagger.TrackCountingHighEffBJetTags
akCs2PFTrackCountingHighPurBJetTags = akCs2PFbTagger.TrackCountingHighPurBJetTags
akCs2PFPatJetPartonAssociationLegacy = akCs2PFbTagger.PatJetPartonAssociationLegacy

akCs2PFImpactParameterTagInfos = akCs2PFbTagger.ImpactParameterTagInfos
akCs2PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs2PFJetProbabilityBJetTags = akCs2PFbTagger.JetProbabilityBJetTags
akCs2PFPositiveOnlyJetProbabilityBJetTags = akCs2PFbTagger.PositiveOnlyJetProbabilityBJetTags
akCs2PFNegativeOnlyJetProbabilityBJetTags = akCs2PFbTagger.NegativeOnlyJetProbabilityBJetTags
akCs2PFNegativeTrackCountingHighEffBJetTags = akCs2PFbTagger.NegativeTrackCountingHighEffBJetTags
akCs2PFNegativeTrackCountingHighPurBJetTags = akCs2PFbTagger.NegativeTrackCountingHighPurBJetTags
akCs2PFNegativeOnlyJetBProbabilityBJetTags = akCs2PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akCs2PFPositiveOnlyJetBProbabilityBJetTags = akCs2PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akCs2PFSecondaryVertexTagInfos = akCs2PFbTagger.SecondaryVertexTagInfos
akCs2PFSimpleSecondaryVertexHighEffBJetTags = akCs2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs2PFSimpleSecondaryVertexHighPurBJetTags = akCs2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs2PFCombinedSecondaryVertexBJetTags = akCs2PFbTagger.CombinedSecondaryVertexBJetTags
akCs2PFCombinedSecondaryVertexV2BJetTags = akCs2PFbTagger.CombinedSecondaryVertexV2BJetTags

akCs2PFSecondaryVertexNegativeTagInfos = akCs2PFbTagger.SecondaryVertexNegativeTagInfos
akCs2PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCs2PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs2PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCs2PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs2PFNegativeCombinedSecondaryVertexBJetTags = akCs2PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCs2PFPositiveCombinedSecondaryVertexBJetTags = akCs2PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akCs2PFSoftPFMuonsTagInfos = akCs2PFbTagger.SoftPFMuonsTagInfos
akCs2PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs2PFSoftPFMuonBJetTags = akCs2PFbTagger.SoftPFMuonBJetTags
akCs2PFSoftPFMuonByIP3dBJetTags = akCs2PFbTagger.SoftPFMuonByIP3dBJetTags
akCs2PFSoftPFMuonByPtBJetTags = akCs2PFbTagger.SoftPFMuonByPtBJetTags
akCs2PFNegativeSoftPFMuonByPtBJetTags = akCs2PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCs2PFPositiveSoftPFMuonByPtBJetTags = akCs2PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCs2PFPatJetFlavourIdLegacy = cms.Sequence(akCs2PFPatJetPartonAssociationLegacy*akCs2PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs2PFPatJetFlavourAssociation = akCs2PFbTagger.PatJetFlavourAssociation
#akCs2PFPatJetFlavourId = cms.Sequence(akCs2PFPatJetPartons*akCs2PFPatJetFlavourAssociation)

akCs2PFJetBtaggingIP       = cms.Sequence(akCs2PFImpactParameterTagInfos *
            (akCs2PFTrackCountingHighEffBJetTags +
             akCs2PFTrackCountingHighPurBJetTags +
             akCs2PFJetProbabilityBJetTags +
             akCs2PFJetBProbabilityBJetTags +
             akCs2PFPositiveOnlyJetProbabilityBJetTags +
             akCs2PFNegativeOnlyJetProbabilityBJetTags +
             akCs2PFNegativeTrackCountingHighEffBJetTags +
             akCs2PFNegativeTrackCountingHighPurBJetTags +
             akCs2PFNegativeOnlyJetBProbabilityBJetTags +
             akCs2PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCs2PFJetBtaggingSV = cms.Sequence(akCs2PFImpactParameterTagInfos
            *
            akCs2PFSecondaryVertexTagInfos
            * (akCs2PFSimpleSecondaryVertexHighEffBJetTags
                +
                akCs2PFSimpleSecondaryVertexHighPurBJetTags
                +
                akCs2PFCombinedSecondaryVertexBJetTags
                +
                akCs2PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCs2PFJetBtaggingNegSV = cms.Sequence(akCs2PFImpactParameterTagInfos
            *
            akCs2PFSecondaryVertexNegativeTagInfos
            * (akCs2PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCs2PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCs2PFNegativeCombinedSecondaryVertexBJetTags
                +
                akCs2PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCs2PFJetBtaggingMu = cms.Sequence(akCs2PFSoftPFMuonsTagInfos * (akCs2PFSoftPFMuonBJetTags
                +
                akCs2PFSoftPFMuonByIP3dBJetTags
                +
                akCs2PFSoftPFMuonByPtBJetTags
                +
                akCs2PFNegativeSoftPFMuonByPtBJetTags
                +
                akCs2PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs2PFJetBtagging = cms.Sequence(akCs2PFJetBtaggingIP
            *akCs2PFJetBtaggingSV
            *akCs2PFJetBtaggingNegSV
#            *akCs2PFJetBtaggingMu
            )

akCs2PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs2PFJets"),
        genJetMatch          = cms.InputTag("akCs2PFmatch"),
        genPartonMatch       = cms.InputTag("akCs2PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs2PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCs2PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs2PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs2PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs2PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs2PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs2PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs2PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs2PFJetBProbabilityBJetTags"),
            cms.InputTag("akCs2PFJetProbabilityBJetTags"),
            #cms.InputTag("akCs2PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs2PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs2PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs2PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs2PFJetID"),
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

akCs2PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs2PFJets"),
           	    R0  = cms.double( 0.2)
)
akCs2PFpatJetsWithBtagging.userData.userFloats.src += ['akCs2PFNjettiness:tau1','akCs2PFNjettiness:tau2','akCs2PFNjettiness:tau3']

akCs2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs2PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCs2PF"),
                                                             jetName = cms.untracked.string("akCs2PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(False)
                                                             )

akCs2PFJetSequence_mc = cms.Sequence(
                                                  #akCs2PFclean
                                                  #*
                                                  akCs2PFmatch
                                                  *
                                                  akCs2PFparton
                                                  *
                                                  akCs2PFcorr
                                                  *
                                                  #akCs2PFJetID
                                                  #*
                                                  akCs2PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCs2PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs2PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCs2PFJetBtagging
                                                  *
                                                  akCs2PFNjettiness
                                                  *
                                                  akCs2PFpatJetsWithBtagging
                                                  *
                                                  akCs2PFJetAnalyzer
                                                  )

akCs2PFJetSequence_data = cms.Sequence(akCs2PFcorr
                                                    *
                                                    #akCs2PFJetID
                                                    #*
                                                    akCs2PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCs2PFJetBtagging
                                                    *
                                                    akCs2PFNjettiness 
                                                    *
                                                    akCs2PFpatJetsWithBtagging
                                                    *
                                                    akCs2PFJetAnalyzer
                                                    )

akCs2PFJetSequence_jec = cms.Sequence(akCs2PFJetSequence_mc)
akCs2PFJetSequence_mb = cms.Sequence(akCs2PFJetSequence_mc)

akCs2PFJetSequence = cms.Sequence(akCs2PFJetSequence_jec)
akCs2PFJetAnalyzer.genPtMin = cms.untracked.double(1)
akCs2PFJetAnalyzer.jetPtMin = cms.double(1)
