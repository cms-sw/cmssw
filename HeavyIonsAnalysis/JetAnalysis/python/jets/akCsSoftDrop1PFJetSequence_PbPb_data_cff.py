

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop1PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop1PFJets"),
    matched = cms.InputTag("ak1HiSignalGenJets"),
    maxDeltaR = 0.1
    )

akCsSoftDrop1PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop1PFJets")
                                                        )

akCsSoftDrop1PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop1PFJets"),
    payload = "AK1PF_offline"
    )

akCsSoftDrop1PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop1CaloJets'))

#akCsSoftDrop1PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiSignalGenJets'))

akCsSoftDrop1PFbTagger = bTaggers("akCsSoftDrop1PF",0.1)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop1PFmatch = akCsSoftDrop1PFbTagger.match
akCsSoftDrop1PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop1PFJets"), matched = cms.InputTag("selectedPartons"))
akCsSoftDrop1PFPatJetFlavourAssociationLegacy = akCsSoftDrop1PFbTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop1PFPatJetPartons = akCsSoftDrop1PFbTagger.PatJetPartons
akCsSoftDrop1PFJetTracksAssociatorAtVertex = akCsSoftDrop1PFbTagger.JetTracksAssociatorAtVertex
akCsSoftDrop1PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop1PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop1PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop1PFCombinedSecondaryVertexBJetTags = akCsSoftDrop1PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop1PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop1PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop1PFJetBProbabilityBJetTags = akCsSoftDrop1PFbTagger.JetBProbabilityBJetTags
akCsSoftDrop1PFSoftPFMuonByPtBJetTags = akCsSoftDrop1PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop1PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop1PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop1PFTrackCountingHighEffBJetTags = akCsSoftDrop1PFbTagger.TrackCountingHighEffBJetTags
akCsSoftDrop1PFTrackCountingHighPurBJetTags = akCsSoftDrop1PFbTagger.TrackCountingHighPurBJetTags
akCsSoftDrop1PFPatJetPartonAssociationLegacy = akCsSoftDrop1PFbTagger.PatJetPartonAssociationLegacy

akCsSoftDrop1PFImpactParameterTagInfos = akCsSoftDrop1PFbTagger.ImpactParameterTagInfos
akCsSoftDrop1PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop1PFJetProbabilityBJetTags = akCsSoftDrop1PFbTagger.JetProbabilityBJetTags

akCsSoftDrop1PFSecondaryVertexTagInfos = akCsSoftDrop1PFbTagger.SecondaryVertexTagInfos
akCsSoftDrop1PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop1PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop1PFCombinedSecondaryVertexBJetTags = akCsSoftDrop1PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop1PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop1PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop1PFSecondaryVertexNegativeTagInfos = akCsSoftDrop1PFbTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop1PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop1PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop1PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop1PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop1PFNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop1PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop1PFPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop1PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsSoftDrop1PFNegativeCombinedSecondaryVertexV2BJetTags = akCsSoftDrop1PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsSoftDrop1PFPositiveCombinedSecondaryVertexV2BJetTags = akCsSoftDrop1PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsSoftDrop1PFSoftPFMuonsTagInfos = akCsSoftDrop1PFbTagger.SoftPFMuonsTagInfos
akCsSoftDrop1PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop1PFSoftPFMuonBJetTags = akCsSoftDrop1PFbTagger.SoftPFMuonBJetTags
akCsSoftDrop1PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop1PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop1PFSoftPFMuonByPtBJetTags = akCsSoftDrop1PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop1PFNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop1PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop1PFPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop1PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop1PFPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop1PFPatJetPartonAssociationLegacy*akCsSoftDrop1PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop1PFPatJetFlavourAssociation = akCsSoftDrop1PFbTagger.PatJetFlavourAssociation
#akCsSoftDrop1PFPatJetFlavourId = cms.Sequence(akCsSoftDrop1PFPatJetPartons*akCsSoftDrop1PFPatJetFlavourAssociation)

akCsSoftDrop1PFJetBtaggingIP       = cms.Sequence(akCsSoftDrop1PFImpactParameterTagInfos *
            (akCsSoftDrop1PFTrackCountingHighEffBJetTags +
             akCsSoftDrop1PFTrackCountingHighPurBJetTags +
             akCsSoftDrop1PFJetProbabilityBJetTags +
             akCsSoftDrop1PFJetBProbabilityBJetTags 
            )
            )

akCsSoftDrop1PFJetBtaggingSV = cms.Sequence(akCsSoftDrop1PFImpactParameterTagInfos
            *
            akCsSoftDrop1PFSecondaryVertexTagInfos
            * (akCsSoftDrop1PFSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop1PFSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop1PFCombinedSecondaryVertexBJetTags+
                akCsSoftDrop1PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop1PFJetBtaggingNegSV = cms.Sequence(akCsSoftDrop1PFImpactParameterTagInfos
            *
            akCsSoftDrop1PFSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop1PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop1PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop1PFNegativeCombinedSecondaryVertexBJetTags+
                akCsSoftDrop1PFPositiveCombinedSecondaryVertexBJetTags+
                akCsSoftDrop1PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsSoftDrop1PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop1PFJetBtaggingMu = cms.Sequence(akCsSoftDrop1PFSoftPFMuonsTagInfos * (akCsSoftDrop1PFSoftPFMuonBJetTags
                +
                akCsSoftDrop1PFSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop1PFSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop1PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop1PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop1PFJetBtagging = cms.Sequence(akCsSoftDrop1PFJetBtaggingIP
            *akCsSoftDrop1PFJetBtaggingSV
            *akCsSoftDrop1PFJetBtaggingNegSV
#            *akCsSoftDrop1PFJetBtaggingMu
            )

akCsSoftDrop1PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop1PFJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop1PFmatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop1PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop1PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop1PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop1PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop1PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop1PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop1PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop1PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop1PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop1PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop1PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop1PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop1PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop1PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop1PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop1PFJetID"),
        addBTagInfo = True,
        addTagInfos = True,
        addDiscriminators = True,
        addAssociatedTracks = True,
        addJetCharge = False,
        addJetID = False,
        getJetMCFlavour = False,
        addGenPartonMatch = False,
        addGenJetMatch = False,
        embedGenJetMatch = False,
        embedGenPartonMatch = False,
        # embedCaloTowers = False,
        # embedPFCandidates = True
        )

akCsSoftDrop1PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop1PFJets"),
           	    R0  = cms.double( 0.1)
)
akCsSoftDrop1PFpatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop1PFNjettiness:tau1','akCsSoftDrop1PFNjettiness:tau2','akCsSoftDrop1PFNjettiness:tau3']

akCsSoftDrop1PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop1PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
                                                             matchJets = cms.untracked.bool(False),
                                                             matchTag = 'patJetsWithBtagging',
                                                             pfCandidateLabel = cms.untracked.InputTag('particleFlowTmp'),
                                                             trackTag = cms.InputTag("hiGeneralTracks"),
                                                             fillGenJets = False,
                                                             isMC = False,
							     doSubEvent = False,
                                                             useHepMC = cms.untracked.bool(False),
							     genParticles = cms.untracked.InputTag("genParticles"),
							     eventInfoTag = cms.InputTag("generator"),
                                                             doLifeTimeTagging = cms.untracked.bool(True),
                                                             doLifeTimeTaggingExtras = cms.untracked.bool(False),
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop1PF"),
                                                             jetName = cms.untracked.string("akCsSoftDrop1PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop1PFJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop1PFclean
                                                  #*
                                                  akCsSoftDrop1PFmatch
                                                  *
                                                  akCsSoftDrop1PFparton
                                                  *
                                                  akCsSoftDrop1PFcorr
                                                  *
                                                  #akCsSoftDrop1PFJetID
                                                  #*
                                                  akCsSoftDrop1PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop1PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop1PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop1PFJetBtagging
                                                  *
                                                  akCsSoftDrop1PFNjettiness
                                                  *
                                                  akCsSoftDrop1PFpatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop1PFJetAnalyzer
                                                  )

akCsSoftDrop1PFJetSequence_data = cms.Sequence(akCsSoftDrop1PFcorr
                                                    *
                                                    #akCsSoftDrop1PFJetID
                                                    #*
                                                    akCsSoftDrop1PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop1PFJetBtagging
                                                    *
                                                    akCsSoftDrop1PFNjettiness 
                                                    *
                                                    akCsSoftDrop1PFpatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop1PFJetAnalyzer
                                                    )

akCsSoftDrop1PFJetSequence_jec = cms.Sequence(akCsSoftDrop1PFJetSequence_mc)
akCsSoftDrop1PFJetSequence_mb = cms.Sequence(akCsSoftDrop1PFJetSequence_mc)

akCsSoftDrop1PFJetSequence = cms.Sequence(akCsSoftDrop1PFJetSequence_data)
