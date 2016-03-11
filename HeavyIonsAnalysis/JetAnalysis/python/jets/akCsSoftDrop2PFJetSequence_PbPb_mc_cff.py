

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop2PFJets"),
    matched = cms.InputTag("ak2HiSignalGenJets"),
    maxDeltaR = 0.2
    )

akCsSoftDrop2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop2PFJets")
                                                        )

akCsSoftDrop2PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop2PFJets"),
    payload = "AK2PF_offline"
    )

akCsSoftDrop2PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop2CaloJets'))

#akCsSoftDrop2PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiSignalGenJets'))

akCsSoftDrop2PFbTagger = bTaggers("akCsSoftDrop2PF",0.2)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop2PFmatch = akCsSoftDrop2PFbTagger.match
akCsSoftDrop2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop2PFJets"), matched = cms.InputTag("selectedPartons"))
akCsSoftDrop2PFPatJetFlavourAssociationLegacy = akCsSoftDrop2PFbTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop2PFPatJetPartons = akCsSoftDrop2PFbTagger.PatJetPartons
akCsSoftDrop2PFJetTracksAssociatorAtVertex = akCsSoftDrop2PFbTagger.JetTracksAssociatorAtVertex
akCsSoftDrop2PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop2PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop2PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop2PFCombinedSecondaryVertexBJetTags = akCsSoftDrop2PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop2PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop2PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop2PFJetBProbabilityBJetTags = akCsSoftDrop2PFbTagger.JetBProbabilityBJetTags
akCsSoftDrop2PFSoftPFMuonByPtBJetTags = akCsSoftDrop2PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop2PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop2PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop2PFTrackCountingHighEffBJetTags = akCsSoftDrop2PFbTagger.TrackCountingHighEffBJetTags
akCsSoftDrop2PFTrackCountingHighPurBJetTags = akCsSoftDrop2PFbTagger.TrackCountingHighPurBJetTags
akCsSoftDrop2PFPatJetPartonAssociationLegacy = akCsSoftDrop2PFbTagger.PatJetPartonAssociationLegacy

akCsSoftDrop2PFImpactParameterTagInfos = akCsSoftDrop2PFbTagger.ImpactParameterTagInfos
akCsSoftDrop2PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop2PFJetProbabilityBJetTags = akCsSoftDrop2PFbTagger.JetProbabilityBJetTags

akCsSoftDrop2PFSecondaryVertexTagInfos = akCsSoftDrop2PFbTagger.SecondaryVertexTagInfos
akCsSoftDrop2PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop2PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop2PFCombinedSecondaryVertexBJetTags = akCsSoftDrop2PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop2PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop2PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop2PFSecondaryVertexNegativeTagInfos = akCsSoftDrop2PFbTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop2PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop2PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop2PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop2PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop2PFNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop2PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop2PFPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop2PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsSoftDrop2PFNegativeCombinedSecondaryVertexV2BJetTags = akCsSoftDrop2PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsSoftDrop2PFPositiveCombinedSecondaryVertexV2BJetTags = akCsSoftDrop2PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsSoftDrop2PFSoftPFMuonsTagInfos = akCsSoftDrop2PFbTagger.SoftPFMuonsTagInfos
akCsSoftDrop2PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop2PFSoftPFMuonBJetTags = akCsSoftDrop2PFbTagger.SoftPFMuonBJetTags
akCsSoftDrop2PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop2PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop2PFSoftPFMuonByPtBJetTags = akCsSoftDrop2PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop2PFNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop2PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop2PFPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop2PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop2PFPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop2PFPatJetPartonAssociationLegacy*akCsSoftDrop2PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop2PFPatJetFlavourAssociation = akCsSoftDrop2PFbTagger.PatJetFlavourAssociation
#akCsSoftDrop2PFPatJetFlavourId = cms.Sequence(akCsSoftDrop2PFPatJetPartons*akCsSoftDrop2PFPatJetFlavourAssociation)

akCsSoftDrop2PFJetBtaggingIP       = cms.Sequence(akCsSoftDrop2PFImpactParameterTagInfos *
            (akCsSoftDrop2PFTrackCountingHighEffBJetTags +
             akCsSoftDrop2PFTrackCountingHighPurBJetTags +
             akCsSoftDrop2PFJetProbabilityBJetTags +
             akCsSoftDrop2PFJetBProbabilityBJetTags 
            )
            )

akCsSoftDrop2PFJetBtaggingSV = cms.Sequence(akCsSoftDrop2PFImpactParameterTagInfos
            *
            akCsSoftDrop2PFSecondaryVertexTagInfos
            * (akCsSoftDrop2PFSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop2PFSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop2PFCombinedSecondaryVertexBJetTags+
                akCsSoftDrop2PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop2PFJetBtaggingNegSV = cms.Sequence(akCsSoftDrop2PFImpactParameterTagInfos
            *
            akCsSoftDrop2PFSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop2PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop2PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop2PFNegativeCombinedSecondaryVertexBJetTags+
                akCsSoftDrop2PFPositiveCombinedSecondaryVertexBJetTags+
                akCsSoftDrop2PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsSoftDrop2PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop2PFJetBtaggingMu = cms.Sequence(akCsSoftDrop2PFSoftPFMuonsTagInfos * (akCsSoftDrop2PFSoftPFMuonBJetTags
                +
                akCsSoftDrop2PFSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop2PFSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop2PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop2PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop2PFJetBtagging = cms.Sequence(akCsSoftDrop2PFJetBtaggingIP
            *akCsSoftDrop2PFJetBtaggingSV
            *akCsSoftDrop2PFJetBtaggingNegSV
#            *akCsSoftDrop2PFJetBtaggingMu
            )

akCsSoftDrop2PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop2PFJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop2PFmatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop2PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop2PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop2PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop2PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop2PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop2PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop2PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop2PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop2PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop2PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop2PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop2PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop2PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop2PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop2PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop2PFJetID"),
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

akCsSoftDrop2PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop2PFJets"),
           	    R0  = cms.double( 0.2)
)
akCsSoftDrop2PFpatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop2PFNjettiness:tau1','akCsSoftDrop2PFNjettiness:tau2','akCsSoftDrop2PFNjettiness:tau3']

akCsSoftDrop2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop2PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop2PF"),
                                                             jetName = cms.untracked.string("akCsSoftDrop2PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop2PFJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop2PFclean
                                                  #*
                                                  akCsSoftDrop2PFmatch
                                                  *
                                                  akCsSoftDrop2PFparton
                                                  *
                                                  akCsSoftDrop2PFcorr
                                                  *
                                                  #akCsSoftDrop2PFJetID
                                                  #*
                                                  akCsSoftDrop2PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop2PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop2PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop2PFJetBtagging
                                                  *
                                                  akCsSoftDrop2PFNjettiness
                                                  *
                                                  akCsSoftDrop2PFpatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop2PFJetAnalyzer
                                                  )

akCsSoftDrop2PFJetSequence_data = cms.Sequence(akCsSoftDrop2PFcorr
                                                    *
                                                    #akCsSoftDrop2PFJetID
                                                    #*
                                                    akCsSoftDrop2PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop2PFJetBtagging
                                                    *
                                                    akCsSoftDrop2PFNjettiness 
                                                    *
                                                    akCsSoftDrop2PFpatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop2PFJetAnalyzer
                                                    )

akCsSoftDrop2PFJetSequence_jec = cms.Sequence(akCsSoftDrop2PFJetSequence_mc)
akCsSoftDrop2PFJetSequence_mb = cms.Sequence(akCsSoftDrop2PFJetSequence_mc)

akCsSoftDrop2PFJetSequence = cms.Sequence(akCsSoftDrop2PFJetSequence_mc)
