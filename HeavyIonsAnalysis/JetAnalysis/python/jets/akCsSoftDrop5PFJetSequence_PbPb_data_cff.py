

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsSoftDrop5PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsSoftDrop5PFJets"),
    matched = cms.InputTag("ak5HiSignalGenJets"),
    maxDeltaR = 0.5
    )

akCsSoftDrop5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop5PFJets")
                                                        )

akCsSoftDrop5PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsSoftDrop5PFJets"),
    payload = "AK5PF_offline"
    )

akCsSoftDrop5PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsSoftDrop5CaloJets'))

#akCsSoftDrop5PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiSignalGenJets'))

akCsSoftDrop5PFbTagger = bTaggers("akCsSoftDrop5PF",0.5)

#create objects locally since they dont load properly otherwise
#akCsSoftDrop5PFmatch = akCsSoftDrop5PFbTagger.match
akCsSoftDrop5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsSoftDrop5PFJets"), matched = cms.InputTag("hiSignalGenParticles"))
akCsSoftDrop5PFPatJetFlavourAssociationLegacy = akCsSoftDrop5PFbTagger.PatJetFlavourAssociationLegacy
akCsSoftDrop5PFPatJetPartons = akCsSoftDrop5PFbTagger.PatJetPartons
akCsSoftDrop5PFJetTracksAssociatorAtVertex = akCsSoftDrop5PFbTagger.JetTracksAssociatorAtVertex
akCsSoftDrop5PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsSoftDrop5PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop5PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop5PFCombinedSecondaryVertexBJetTags = akCsSoftDrop5PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop5PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop5PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsSoftDrop5PFJetBProbabilityBJetTags = akCsSoftDrop5PFbTagger.JetBProbabilityBJetTags
akCsSoftDrop5PFSoftPFMuonByPtBJetTags = akCsSoftDrop5PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop5PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop5PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop5PFTrackCountingHighEffBJetTags = akCsSoftDrop5PFbTagger.TrackCountingHighEffBJetTags
akCsSoftDrop5PFTrackCountingHighPurBJetTags = akCsSoftDrop5PFbTagger.TrackCountingHighPurBJetTags
akCsSoftDrop5PFPatJetPartonAssociationLegacy = akCsSoftDrop5PFbTagger.PatJetPartonAssociationLegacy

akCsSoftDrop5PFImpactParameterTagInfos = akCsSoftDrop5PFbTagger.ImpactParameterTagInfos
akCsSoftDrop5PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop5PFJetProbabilityBJetTags = akCsSoftDrop5PFbTagger.JetProbabilityBJetTags

akCsSoftDrop5PFSecondaryVertexTagInfos = akCsSoftDrop5PFbTagger.SecondaryVertexTagInfos
akCsSoftDrop5PFSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop5PFSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop5PFCombinedSecondaryVertexBJetTags = akCsSoftDrop5PFbTagger.CombinedSecondaryVertexBJetTags
akCsSoftDrop5PFCombinedSecondaryVertexV2BJetTags = akCsSoftDrop5PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsSoftDrop5PFSecondaryVertexNegativeTagInfos = akCsSoftDrop5PFbTagger.SecondaryVertexNegativeTagInfos
akCsSoftDrop5PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsSoftDrop5PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsSoftDrop5PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsSoftDrop5PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsSoftDrop5PFNegativeCombinedSecondaryVertexBJetTags = akCsSoftDrop5PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsSoftDrop5PFPositiveCombinedSecondaryVertexBJetTags = akCsSoftDrop5PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsSoftDrop5PFNegativeCombinedSecondaryVertexV2BJetTags = akCsSoftDrop5PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsSoftDrop5PFPositiveCombinedSecondaryVertexV2BJetTags = akCsSoftDrop5PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsSoftDrop5PFSoftPFMuonsTagInfos = akCsSoftDrop5PFbTagger.SoftPFMuonsTagInfos
akCsSoftDrop5PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsSoftDrop5PFSoftPFMuonBJetTags = akCsSoftDrop5PFbTagger.SoftPFMuonBJetTags
akCsSoftDrop5PFSoftPFMuonByIP3dBJetTags = akCsSoftDrop5PFbTagger.SoftPFMuonByIP3dBJetTags
akCsSoftDrop5PFSoftPFMuonByPtBJetTags = akCsSoftDrop5PFbTagger.SoftPFMuonByPtBJetTags
akCsSoftDrop5PFNegativeSoftPFMuonByPtBJetTags = akCsSoftDrop5PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsSoftDrop5PFPositiveSoftPFMuonByPtBJetTags = akCsSoftDrop5PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsSoftDrop5PFPatJetFlavourIdLegacy = cms.Sequence(akCsSoftDrop5PFPatJetPartonAssociationLegacy*akCsSoftDrop5PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsSoftDrop5PFPatJetFlavourAssociation = akCsSoftDrop5PFbTagger.PatJetFlavourAssociation
#akCsSoftDrop5PFPatJetFlavourId = cms.Sequence(akCsSoftDrop5PFPatJetPartons*akCsSoftDrop5PFPatJetFlavourAssociation)

akCsSoftDrop5PFJetBtaggingIP       = cms.Sequence(akCsSoftDrop5PFImpactParameterTagInfos *
            (akCsSoftDrop5PFTrackCountingHighEffBJetTags +
             akCsSoftDrop5PFTrackCountingHighPurBJetTags +
             akCsSoftDrop5PFJetProbabilityBJetTags +
             akCsSoftDrop5PFJetBProbabilityBJetTags 
            )
            )

akCsSoftDrop5PFJetBtaggingSV = cms.Sequence(akCsSoftDrop5PFImpactParameterTagInfos
            *
            akCsSoftDrop5PFSecondaryVertexTagInfos
            * (akCsSoftDrop5PFSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop5PFSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop5PFCombinedSecondaryVertexBJetTags+
                akCsSoftDrop5PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop5PFJetBtaggingNegSV = cms.Sequence(akCsSoftDrop5PFImpactParameterTagInfos
            *
            akCsSoftDrop5PFSecondaryVertexNegativeTagInfos
            * (akCsSoftDrop5PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsSoftDrop5PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsSoftDrop5PFNegativeCombinedSecondaryVertexBJetTags+
                akCsSoftDrop5PFPositiveCombinedSecondaryVertexBJetTags+
                akCsSoftDrop5PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsSoftDrop5PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsSoftDrop5PFJetBtaggingMu = cms.Sequence(akCsSoftDrop5PFSoftPFMuonsTagInfos * (akCsSoftDrop5PFSoftPFMuonBJetTags
                +
                akCsSoftDrop5PFSoftPFMuonByIP3dBJetTags
                +
                akCsSoftDrop5PFSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop5PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsSoftDrop5PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsSoftDrop5PFJetBtagging = cms.Sequence(akCsSoftDrop5PFJetBtaggingIP
            *akCsSoftDrop5PFJetBtaggingSV
            *akCsSoftDrop5PFJetBtaggingNegSV
#            *akCsSoftDrop5PFJetBtaggingMu
            )

akCsSoftDrop5PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsSoftDrop5PFJets"),
        genJetMatch          = cms.InputTag("akCsSoftDrop5PFmatch"),
        genPartonMatch       = cms.InputTag("akCsSoftDrop5PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsSoftDrop5PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsSoftDrop5PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsSoftDrop5PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsSoftDrop5PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsSoftDrop5PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop5PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsSoftDrop5PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsSoftDrop5PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsSoftDrop5PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsSoftDrop5PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsSoftDrop5PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsSoftDrop5PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsSoftDrop5PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsSoftDrop5PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsSoftDrop5PFJetID"),
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

akCsSoftDrop5PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsSoftDrop5PFJets"),
           	    R0  = cms.double( 0.5)
)
akCsSoftDrop5PFpatJetsWithBtagging.userData.userFloats.src += ['akCsSoftDrop5PFNjettiness:tau1','akCsSoftDrop5PFNjettiness:tau2','akCsSoftDrop5PFNjettiness:tau3']

akCsSoftDrop5PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsSoftDrop5PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak5HiGenJets',
                                                             rParam = 0.5,
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
                                                             bTagJetName = cms.untracked.string("akCsSoftDrop5PF"),
                                                             jetName = cms.untracked.string("akCsSoftDrop5PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsSoftDrop5PFJetSequence_mc = cms.Sequence(
                                                  #akCsSoftDrop5PFclean
                                                  #*
                                                  akCsSoftDrop5PFmatch
                                                  *
                                                  akCsSoftDrop5PFparton
                                                  *
                                                  akCsSoftDrop5PFcorr
                                                  *
                                                  #akCsSoftDrop5PFJetID
                                                  #*
                                                  akCsSoftDrop5PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsSoftDrop5PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsSoftDrop5PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsSoftDrop5PFJetBtagging
                                                  *
                                                  akCsSoftDrop5PFNjettiness
                                                  *
                                                  akCsSoftDrop5PFpatJetsWithBtagging
                                                  *
                                                  akCsSoftDrop5PFJetAnalyzer
                                                  )

akCsSoftDrop5PFJetSequence_data = cms.Sequence(akCsSoftDrop5PFcorr
                                                    *
                                                    #akCsSoftDrop5PFJetID
                                                    #*
                                                    akCsSoftDrop5PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsSoftDrop5PFJetBtagging
                                                    *
                                                    akCsSoftDrop5PFNjettiness 
                                                    *
                                                    akCsSoftDrop5PFpatJetsWithBtagging
                                                    *
                                                    akCsSoftDrop5PFJetAnalyzer
                                                    )

akCsSoftDrop5PFJetSequence_jec = cms.Sequence(akCsSoftDrop5PFJetSequence_mc)
akCsSoftDrop5PFJetSequence_mb = cms.Sequence(akCsSoftDrop5PFJetSequence_mc)

akCsSoftDrop5PFJetSequence = cms.Sequence(akCsSoftDrop5PFJetSequence_data)
