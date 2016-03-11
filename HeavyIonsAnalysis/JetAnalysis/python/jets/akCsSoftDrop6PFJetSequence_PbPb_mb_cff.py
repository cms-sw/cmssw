

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop6PFJets"),
    matched = cms.InputTag("ak6HiCleanedGenJets"),
    maxDeltaR = 0.6
    )

akCsSoftDrop6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop6PFJets")
                                                        )

akCsSoftDrop6PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop6PFJets"),
    payload = "AK6PF_offline"
    )

akCsSoftDrop6PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop6CaloJets'))

#akCsSoftDrop6PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiCleanedGenJets'))

akCsSoftDrop6PFbTagger = bTaggers("akCsSoftDrop6PF",0.6)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop6PFmatch = akCsSoftDrop6PFbTagger.match
akCsSoftDrop6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop6PFJets"), matched = cms.InputTag("selectedPartons"))
akCsSoftDrop6PFPatJetFlavourAssociationLegacy = akCsSoftDrop6PFbTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop6PFPatJetPartons = akCsSoftDrop6PFbTagger.PatJetPartons
akCsSoftDrop6PFJetTracksAssociatorAtVertex = akCsSoftDrop6PFbTagger.JetTracksAssociatorAtVertex
akCsSoftDrop6PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop6PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop6PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop6PFCombinedSecondaryVertexBJetTags = akCsSoftDrop6PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop6PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop6PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop6PFJetBProbabilityBJetTags = akCsSoftDrop6PFbTagger.JetBProbabilityBJetTags
akCsSoftDrop6PFSoftPFMuonByPtBJetTags = akCsSoftDrop6PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop6PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop6PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop6PFTrackCountingHighEffBJetTags = akCsSoftDrop6PFbTagger.TrackCountingHighEffBJetTags
akCsSoftDrop6PFTrackCountingHighPurBJetTags = akCsSoftDrop6PFbTagger.TrackCountingHighPurBJetTags
akCsSoftDrop6PFPatJetPartonAssociationLegacy = akCsSoftDrop6PFbTagger.PatJetPartonAssociationLegacy

akCsSoftDrop6PFImpactParameterTagInfos = akCsSoftDrop6PFbTagger.ImpactParameterTagInfos
akCsSoftDrop6PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop6PFJetProbabilityBJetTags = akCsSoftDrop6PFbTagger.JetProbabilityBJetTags
akCsSoftDrop6PFPositiveOnlyJetProbabilityBJetTags = akCsSoftDrop6PFbTagger.PositiveOnlyJetProbabilityBJetTags
akCsSoftDrop6PFNegativeOnlyJetProbabilityBJetTags = akCsSoftDrop6PFbTagger.NegativeOnlyJetProbabilityBJetTags
akCsSoftDrop6PFNegativeTrackCountingHighEffBJetTags = akCsSoftDrop6PFbTagger.NegativeTrackCountingHighEffBJetTags
akCsSoftDrop6PFNegativeTrackCountingHighPurBJetTags = akCsSoftDrop6PFbTagger.NegativeTrackCountingHighPurBJetTags
akCsSoftDrop6PFNegativeOnlyJetBProbabilityBJetTags = akCsSoftDrop6PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akCsSoftDrop6PFPositiveOnlyJetBProbabilityBJetTags = akCsSoftDrop6PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akCsSoftDrop6PFSecondaryVertexTagInfos = akCsSoftDrop6PFbTagger.SecondaryVertexTagInfos
akCsSoftDrop6PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop6PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop6PFCombinedSecondaryVertexBJetTags = akCsSoftDrop6PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop6PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop6PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop6PFSecondaryVertexNegativeTagInfos = akCsSoftDrop6PFbTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop6PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop6PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop6PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop6PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop6PFNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop6PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop6PFPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop6PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akCsSoftDrop6PFSoftPFMuonsTagInfos = akCsSoftDrop6PFbTagger.SoftPFMuonsTagInfos
akCsSoftDrop6PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop6PFSoftPFMuonBJetTags = akCsSoftDrop6PFbTagger.SoftPFMuonBJetTags
akCsSoftDrop6PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop6PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop6PFSoftPFMuonByPtBJetTags = akCsSoftDrop6PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop6PFNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop6PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop6PFPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop6PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop6PFPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop6PFPatJetPartonAssociationLegacy*akCsSoftDrop6PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop6PFPatJetFlavourAssociation = akCsSoftDrop6PFbTagger.PatJetFlavourAssociation
#akCsSoftDrop6PFPatJetFlavourId = cms.Sequence(akCsSoftDrop6PFPatJetPartons*akCsSoftDrop6PFPatJetFlavourAssociation)

akCsSoftDrop6PFJetBtaggingIP       = cms.Sequence(akCsSoftDrop6PFImpactParameterTagInfos *
            (akCsSoftDrop6PFTrackCountingHighEffBJetTags +
             akCsSoftDrop6PFTrackCountingHighPurBJetTags +
             akCsSoftDrop6PFJetProbabilityBJetTags +
             akCsSoftDrop6PFJetBProbabilityBJetTags +
             akCsSoftDrop6PFPositiveOnlyJetProbabilityBJetTags +
             akCsSoftDrop6PFNegativeOnlyJetProbabilityBJetTags +
             akCsSoftDrop6PFNegativeTrackCountingHighEffBJetTags +
             akCsSoftDrop6PFNegativeTrackCountingHighPurBJetTags +
             akCsSoftDrop6PFNegativeOnlyJetBProbabilityBJetTags +
             akCsSoftDrop6PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akCsSoftDrop6PFJetBtaggingSV = cms.Sequence(akCsSoftDrop6PFImpactParameterTagInfos
            *
            akCsSoftDrop6PFSecondaryVertexTagInfos
            * (akCsSoftDrop6PFSimpleSecondaryVertexHighEffBJetTags
                +
                akCsSoftDrop6PFSimpleSecondaryVertexHighPurBJetTags
                +
                akCsSoftDrop6PFCombinedSecondaryVertexBJetTags
                +
                akCsSoftDrop6PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop6PFJetBtaggingNegSV = cms.Sequence(akCsSoftDrop6PFImpactParameterTagInfos
            *
            akCsSoftDrop6PFSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop6PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akCsSoftDrop6PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akCsSoftDrop6PFNegativeCombinedSecondaryVertexBJetTags
                +
                akCsSoftDrop6PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akCsSoftDrop6PFJetBtaggingMu = cms.Sequence(akCsSoftDrop6PFSoftPFMuonsTagInfos * (akCsSoftDrop6PFSoftPFMuonBJetTags
                +
                akCsSoftDrop6PFSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop6PFSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop6PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop6PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop6PFJetBtagging = cms.Sequence(akCsSoftDrop6PFJetBtaggingIP
            *akCsSoftDrop6PFJetBtaggingSV
            *akCsSoftDrop6PFJetBtaggingNegSV
#            *akCsSoftDrop6PFJetBtaggingMu
            )

akCsSoftDrop6PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop6PFJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop6PFmatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop6PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop6PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop6PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop6PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop6PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop6PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop6PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop6PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop6PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop6PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop6PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop6PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop6PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop6PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop6PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop6PFJetID"),
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

akCsSoftDrop6PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop6PFJets"),
           	    R0  = cms.double( 0.6)
)
akCsSoftDrop6PFpatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop6PFNjettiness:tau1','akCsSoftDrop6PFNjettiness:tau2','akCsSoftDrop6PFNjettiness:tau3']

akCsSoftDrop6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop6PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop6PF"),
                                                             jetName = cms.untracked.string("akCsSoftDrop6PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop6PFJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop6PFclean
                                                  #*
                                                  akCsSoftDrop6PFmatch
                                                  *
                                                  akCsSoftDrop6PFparton
                                                  *
                                                  akCsSoftDrop6PFcorr
                                                  *
                                                  #akCsSoftDrop6PFJetID
                                                  #*
                                                  akCsSoftDrop6PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop6PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop6PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop6PFJetBtagging
                                                  *
                                                  akCsSoftDrop6PFNjettiness
                                                  *
                                                  akCsSoftDrop6PFpatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop6PFJetAnalyzer
                                                  )

akCsSoftDrop6PFJetSequence_data = cms.Sequence(akCsSoftDrop6PFcorr
                                                    *
                                                    #akCsSoftDrop6PFJetID
                                                    #*
                                                    akCsSoftDrop6PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop6PFJetBtagging
                                                    *
                                                    akCsSoftDrop6PFNjettiness 
                                                    *
                                                    akCsSoftDrop6PFpatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop6PFJetAnalyzer
                                                    )

akCsSoftDrop6PFJetSequence_jec = cms.Sequence(akCsSoftDrop6PFJetSequence_mc)
akCsSoftDrop6PFJetSequence_mb = cms.Sequence(akCsSoftDrop6PFJetSequence_mc)

akCsSoftDrop6PFJetSequence = cms.Sequence(akCsSoftDrop6PFJetSequence_mb)
