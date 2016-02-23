

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akPu4PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu4PFJets"),
    matched = cms.InputTag("ak4HiGenJets"),
    maxDeltaR = 0.4
    )

akPu4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu4PFJets")
                                                        )

akPu4PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu4PFJets"),
    payload = "AK4PF_offline"
    )

akPu4PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu4CaloJets'))

#akPu4PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak4HiGenJets'))

akPu4PFbTagger = bTaggers("akPu4PF",0.4)

#create objects locally since they dont load properly otherwise
#akPu4PFmatch = akPu4PFbTagger.match
akPu4PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu4PFJets"), matched = cms.InputTag("genParticles"))
akPu4PFPatJetFlavourAssociationLegacy = akPu4PFbTagger.PatJetFlavourAssociationLegacy
akPu4PFPatJetPartons = akPu4PFbTagger.PatJetPartons
akPu4PFJetTracksAssociatorAtVertex = akPu4PFbTagger.JetTracksAssociatorAtVertex
akPu4PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akPu4PFSimpleSecondaryVertexHighEffBJetTags = akPu4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu4PFSimpleSecondaryVertexHighPurBJetTags = akPu4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu4PFCombinedSecondaryVertexBJetTags = akPu4PFbTagger.CombinedSecondaryVertexBJetTags
akPu4PFCombinedSecondaryVertexV2BJetTags = akPu4PFbTagger.CombinedSecondaryVertexV2BJetTags
akPu4PFJetBProbabilityBJetTags = akPu4PFbTagger.JetBProbabilityBJetTags
akPu4PFSoftPFMuonByPtBJetTags = akPu4PFbTagger.SoftPFMuonByPtBJetTags
akPu4PFSoftPFMuonByIP3dBJetTags = akPu4PFbTagger.SoftPFMuonByIP3dBJetTags
akPu4PFTrackCountingHighEffBJetTags = akPu4PFbTagger.TrackCountingHighEffBJetTags
akPu4PFTrackCountingHighPurBJetTags = akPu4PFbTagger.TrackCountingHighPurBJetTags
akPu4PFPatJetPartonAssociationLegacy = akPu4PFbTagger.PatJetPartonAssociationLegacy

akPu4PFImpactParameterTagInfos = akPu4PFbTagger.ImpactParameterTagInfos
akPu4PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu4PFJetProbabilityBJetTags = akPu4PFbTagger.JetProbabilityBJetTags
akPu4PFPositiveOnlyJetProbabilityBJetTags = akPu4PFbTagger.PositiveOnlyJetProbabilityBJetTags
akPu4PFNegativeOnlyJetProbabilityBJetTags = akPu4PFbTagger.NegativeOnlyJetProbabilityBJetTags
akPu4PFNegativeTrackCountingHighEffBJetTags = akPu4PFbTagger.NegativeTrackCountingHighEffBJetTags
akPu4PFNegativeTrackCountingHighPurBJetTags = akPu4PFbTagger.NegativeTrackCountingHighPurBJetTags
akPu4PFNegativeOnlyJetBProbabilityBJetTags = akPu4PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akPu4PFPositiveOnlyJetBProbabilityBJetTags = akPu4PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akPu4PFSecondaryVertexTagInfos = akPu4PFbTagger.SecondaryVertexTagInfos
akPu4PFSimpleSecondaryVertexHighEffBJetTags = akPu4PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu4PFSimpleSecondaryVertexHighPurBJetTags = akPu4PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu4PFCombinedSecondaryVertexBJetTags = akPu4PFbTagger.CombinedSecondaryVertexBJetTags
akPu4PFCombinedSecondaryVertexV2BJetTags = akPu4PFbTagger.CombinedSecondaryVertexV2BJetTags

akPu4PFSecondaryVertexNegativeTagInfos = akPu4PFbTagger.SecondaryVertexNegativeTagInfos
akPu4PFNegativeSimpleSecondaryVertexHighEffBJetTags = akPu4PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akPu4PFNegativeSimpleSecondaryVertexHighPurBJetTags = akPu4PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akPu4PFNegativeCombinedSecondaryVertexBJetTags = akPu4PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akPu4PFPositiveCombinedSecondaryVertexBJetTags = akPu4PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akPu4PFSoftPFMuonsTagInfos = akPu4PFbTagger.SoftPFMuonsTagInfos
akPu4PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu4PFSoftPFMuonBJetTags = akPu4PFbTagger.SoftPFMuonBJetTags
akPu4PFSoftPFMuonByIP3dBJetTags = akPu4PFbTagger.SoftPFMuonByIP3dBJetTags
akPu4PFSoftPFMuonByPtBJetTags = akPu4PFbTagger.SoftPFMuonByPtBJetTags
akPu4PFNegativeSoftPFMuonByPtBJetTags = akPu4PFbTagger.NegativeSoftPFMuonByPtBJetTags
akPu4PFPositiveSoftPFMuonByPtBJetTags = akPu4PFbTagger.PositiveSoftPFMuonByPtBJetTags
akPu4PFPatJetFlavourIdLegacy = cms.Sequence(akPu4PFPatJetPartonAssociationLegacy*akPu4PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akPu4PFPatJetFlavourAssociation = akPu4PFbTagger.PatJetFlavourAssociation
#akPu4PFPatJetFlavourId = cms.Sequence(akPu4PFPatJetPartons*akPu4PFPatJetFlavourAssociation)

akPu4PFJetBtaggingIP       = cms.Sequence(akPu4PFImpactParameterTagInfos *
            (akPu4PFTrackCountingHighEffBJetTags +
             akPu4PFTrackCountingHighPurBJetTags +
             akPu4PFJetProbabilityBJetTags +
             akPu4PFJetBProbabilityBJetTags +
             akPu4PFPositiveOnlyJetProbabilityBJetTags +
             akPu4PFNegativeOnlyJetProbabilityBJetTags +
             akPu4PFNegativeTrackCountingHighEffBJetTags +
             akPu4PFNegativeTrackCountingHighPurBJetTags +
             akPu4PFNegativeOnlyJetBProbabilityBJetTags +
             akPu4PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akPu4PFJetBtaggingSV = cms.Sequence(akPu4PFImpactParameterTagInfos
            *
            akPu4PFSecondaryVertexTagInfos
            * (akPu4PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu4PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu4PFCombinedSecondaryVertexBJetTags
                +
                akPu4PFCombinedSecondaryVertexV2BJetTags
              )
            )

akPu4PFJetBtaggingNegSV = cms.Sequence(akPu4PFImpactParameterTagInfos
            *
            akPu4PFSecondaryVertexNegativeTagInfos
            * (akPu4PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akPu4PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akPu4PFNegativeCombinedSecondaryVertexBJetTags
                +
                akPu4PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akPu4PFJetBtaggingMu = cms.Sequence(akPu4PFSoftPFMuonsTagInfos * (akPu4PFSoftPFMuonBJetTags
                +
                akPu4PFSoftPFMuonByIP3dBJetTags
                +
                akPu4PFSoftPFMuonByPtBJetTags
                +
                akPu4PFNegativeSoftPFMuonByPtBJetTags
                +
                akPu4PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akPu4PFJetBtagging = cms.Sequence(akPu4PFJetBtaggingIP
            *akPu4PFJetBtaggingSV
            *akPu4PFJetBtaggingNegSV
#            *akPu4PFJetBtaggingMu
            )

akPu4PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu4PFJets"),
        genJetMatch          = cms.InputTag("akPu4PFmatch"),
        genPartonMatch       = cms.InputTag("akPu4PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu4PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu4PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akPu4PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu4PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu4PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu4PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu4PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu4PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akPu4PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu4PFJetProbabilityBJetTags"),
            #cms.InputTag("akPu4PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akPu4PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akPu4PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu4PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu4PFJetID"),
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

akPu4PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akPu4PFJets"),
           	    R0  = cms.double( 0.4)
)
akPu4PFpatJetsWithBtagging.userData.userFloats.src += ['akPu4PFNjettiness:tau1','akPu4PFNjettiness:tau2','akPu4PFNjettiness:tau3']

akPu4PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu4PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akPu4PF"),
                                                             jetName = cms.untracked.string("akPu4PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akPu4PFJetSequence_mc = cms.Sequence(
                                                  #akPu4PFclean
                                                  #*
                                                  akPu4PFmatch
                                                  *
                                                  akPu4PFparton
                                                  *
                                                  akPu4PFcorr
                                                  *
                                                  #akPu4PFJetID
                                                  #*
                                                  akPu4PFPatJetFlavourIdLegacy
                                                  #*
			                          #akPu4PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akPu4PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu4PFJetBtagging
                                                  *
                                                  akPu4PFNjettiness
                                                  *
                                                  akPu4PFpatJetsWithBtagging
                                                  *
                                                  akPu4PFJetAnalyzer
                                                  )

akPu4PFJetSequence_data = cms.Sequence(akPu4PFcorr
                                                    *
                                                    #akPu4PFJetID
                                                    #*
                                                    akPu4PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu4PFJetBtagging
                                                    *
                                                    akPu4PFNjettiness 
                                                    *
                                                    akPu4PFpatJetsWithBtagging
                                                    *
                                                    akPu4PFJetAnalyzer
                                                    )

akPu4PFJetSequence_jec = cms.Sequence(akPu4PFJetSequence_mc)
akPu4PFJetSequence_mix = cms.Sequence(akPu4PFJetSequence_mc)

akPu4PFJetSequence = cms.Sequence(akPu4PFJetSequence_mc)
