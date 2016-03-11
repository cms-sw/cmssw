

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop4PFJets"),
    matched = cms.InputTag("ak4HiSignalGenJets"),
    maxDeltaR = 0.4
    )

akCsSoftDrop4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop4PFJets")
                                                        )

akCsSoftDrop4PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop4PFJets"),
    payload = "AK4PF_offline"
    )

akCsSoftDrop4PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop4CaloJets'))

#akCsSoftDrop4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiSignalGenJets'))

akCsSoftDrop4PFbTagger = bTaggers("akCsSoftDrop4PF",0.4)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop4PFmatch = akCsSoftDrop4PFbTagger.match
akCsSoftDrop4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop4PFJets"), matched = cms.InputTag("selectedPartons"))
akCsSoftDrop4PFPatJetFlavourAssociationLegacy = akCsSoftDrop4PFbTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop4PFPatJetPartons = akCsSoftDrop4PFbTagger.PatJetPartons
akCsSoftDrop4PFJetTracksAssociatorAtVertex = akCsSoftDrop4PFbTagger.JetTracksAssociatorAtVertex
akCsSoftDrop4PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop4PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop4PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop4PFCombinedSecondaryVertexBJetTags = akCsSoftDrop4PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop4PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop4PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop4PFJetBProbabilityBJetTags = akCsSoftDrop4PFbTagger.JetBProbabilityBJetTags
akCsSoftDrop4PFSoftPFMuonByPtBJetTags = akCsSoftDrop4PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop4PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop4PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop4PFTrackCountingHighEffBJetTags = akCsSoftDrop4PFbTagger.TrackCountingHighEffBJetTags
akCsSoftDrop4PFTrackCountingHighPurBJetTags = akCsSoftDrop4PFbTagger.TrackCountingHighPurBJetTags
akCsSoftDrop4PFPatJetPartonAssociationLegacy = akCsSoftDrop4PFbTagger.PatJetPartonAssociationLegacy

akCsSoftDrop4PFImpactParameterTagInfos = akCsSoftDrop4PFbTagger.ImpactParameterTagInfos
akCsSoftDrop4PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop4PFJetProbabilityBJetTags = akCsSoftDrop4PFbTagger.JetProbabilityBJetTags

akCsSoftDrop4PFSecondaryVertexTagInfos = akCsSoftDrop4PFbTagger.SecondaryVertexTagInfos
akCsSoftDrop4PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop4PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop4PFCombinedSecondaryVertexBJetTags = akCsSoftDrop4PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop4PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop4PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop4PFSecondaryVertexNegativeTagInfos = akCsSoftDrop4PFbTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop4PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop4PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop4PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop4PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop4PFNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop4PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop4PFPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop4PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsSoftDrop4PFNegativeCombinedSecondaryVertexV2BJetTags = akCsSoftDrop4PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsSoftDrop4PFPositiveCombinedSecondaryVertexV2BJetTags = akCsSoftDrop4PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsSoftDrop4PFSoftPFMuonsTagInfos = akCsSoftDrop4PFbTagger.SoftPFMuonsTagInfos
akCsSoftDrop4PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop4PFSoftPFMuonBJetTags = akCsSoftDrop4PFbTagger.SoftPFMuonBJetTags
akCsSoftDrop4PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop4PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop4PFSoftPFMuonByPtBJetTags = akCsSoftDrop4PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop4PFNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop4PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop4PFPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop4PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop4PFPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop4PFPatJetPartonAssociationLegacy*akCsSoftDrop4PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop4PFPatJetFlavourAssociation = akCsSoftDrop4PFbTagger.PatJetFlavourAssociation
#akCsSoftDrop4PFPatJetFlavourId = cms.Sequence(akCsSoftDrop4PFPatJetPartons*akCsSoftDrop4PFPatJetFlavourAssociation)

akCsSoftDrop4PFJetBtaggingIP       = cms.Sequence(akCsSoftDrop4PFImpactParameterTagInfos *
            (akCsSoftDrop4PFTrackCountingHighEffBJetTags +
             akCsSoftDrop4PFTrackCountingHighPurBJetTags +
             akCsSoftDrop4PFJetProbabilityBJetTags +
             akCsSoftDrop4PFJetBProbabilityBJetTags 
            )
            )

akCsSoftDrop4PFJetBtaggingSV = cms.Sequence(akCsSoftDrop4PFImpactParameterTagInfos
            *
            akCsSoftDrop4PFSecondaryVertexTagInfos
            * (akCsSoftDrop4PFSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop4PFSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop4PFCombinedSecondaryVertexBJetTags+
                akCsSoftDrop4PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop4PFJetBtaggingNegSV = cms.Sequence(akCsSoftDrop4PFImpactParameterTagInfos
            *
            akCsSoftDrop4PFSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop4PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop4PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop4PFNegativeCombinedSecondaryVertexBJetTags+
                akCsSoftDrop4PFPositiveCombinedSecondaryVertexBJetTags+
                akCsSoftDrop4PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsSoftDrop4PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop4PFJetBtaggingMu = cms.Sequence(akCsSoftDrop4PFSoftPFMuonsTagInfos * (akCsSoftDrop4PFSoftPFMuonBJetTags
                +
                akCsSoftDrop4PFSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop4PFSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop4PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop4PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop4PFJetBtagging = cms.Sequence(akCsSoftDrop4PFJetBtaggingIP
            *akCsSoftDrop4PFJetBtaggingSV
            *akCsSoftDrop4PFJetBtaggingNegSV
#            *akCsSoftDrop4PFJetBtaggingMu
            )

akCsSoftDrop4PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop4PFJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop4PFmatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop4PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop4PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop4PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop4PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop4PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop4PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop4PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop4PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop4PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop4PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop4PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop4PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop4PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop4PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop4PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop4PFJetID"),
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

akCsSoftDrop4PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop4PFJets"),
           	    R0  = cms.double( 0.4)
)
akCsSoftDrop4PFpatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop4PFNjettiness:tau1','akCsSoftDrop4PFNjettiness:tau2','akCsSoftDrop4PFNjettiness:tau3']

akCsSoftDrop4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop4PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop4PF"),
                                                             jetName = cms.untracked.string("akCsSoftDrop4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop4PFJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop4PFclean
                                                  #*
                                                  akCsSoftDrop4PFmatch
                                                  *
                                                  akCsSoftDrop4PFparton
                                                  *
                                                  akCsSoftDrop4PFcorr
                                                  *
                                                  #akCsSoftDrop4PFJetID
                                                  #*
                                                  akCsSoftDrop4PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop4PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop4PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop4PFJetBtagging
                                                  *
                                                  akCsSoftDrop4PFNjettiness
                                                  *
                                                  akCsSoftDrop4PFpatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop4PFJetAnalyzer
                                                  )

akCsSoftDrop4PFJetSequence_data = cms.Sequence(akCsSoftDrop4PFcorr
                                                    *
                                                    #akCsSoftDrop4PFJetID
                                                    #*
                                                    akCsSoftDrop4PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop4PFJetBtagging
                                                    *
                                                    akCsSoftDrop4PFNjettiness 
                                                    *
                                                    akCsSoftDrop4PFpatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop4PFJetAnalyzer
                                                    )

akCsSoftDrop4PFJetSequence_jec = cms.Sequence(akCsSoftDrop4PFJetSequence_mc)
akCsSoftDrop4PFJetSequence_mb = cms.Sequence(akCsSoftDrop4PFJetSequence_mc)

akCsSoftDrop4PFJetSequence = cms.Sequence(akCsSoftDrop4PFJetSequence_mc)
