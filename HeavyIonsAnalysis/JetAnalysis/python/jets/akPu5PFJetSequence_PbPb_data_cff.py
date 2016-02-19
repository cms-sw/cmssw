

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akPu5PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akPu5PFJets"),
    matched = cms.InputTag("ak5HiGenJets"),
    maxDeltaR = 0.5
    )

akPu5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu5PFJets")
                                                        )

akPu5PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akPu5PFJets"),
    payload = "AK5PF_offline"
    )

akPu5PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akPu5CaloJets'))

#akPu5PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiGenJets'))

akPu5PFbTagger = bTaggers("akPu5PF",0.5)

#create objects locally since they dont load properly otherwise
#akPu5PFmatch = akPu5PFbTagger.match
akPu5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akPu5PFJets"), matched = cms.InputTag("genParticles"))
akPu5PFPatJetFlavourAssociationLegacy = akPu5PFbTagger.PatJetFlavourAssociationLegacy
akPu5PFPatJetPartons = akPu5PFbTagger.PatJetPartons
akPu5PFJetTracksAssociatorAtVertex = akPu5PFbTagger.JetTracksAssociatorAtVertex
akPu5PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akPu5PFSimpleSecondaryVertexHighEffBJetTags = akPu5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu5PFSimpleSecondaryVertexHighPurBJetTags = akPu5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu5PFCombinedSecondaryVertexBJetTags = akPu5PFbTagger.CombinedSecondaryVertexBJetTags
akPu5PFCombinedSecondaryVertexV2BJetTags = akPu5PFbTagger.CombinedSecondaryVertexV2BJetTags
akPu5PFJetBProbabilityBJetTags = akPu5PFbTagger.JetBProbabilityBJetTags
akPu5PFSoftPFMuonByPtBJetTags = akPu5PFbTagger.SoftPFMuonByPtBJetTags
akPu5PFSoftPFMuonByIP3dBJetTags = akPu5PFbTagger.SoftPFMuonByIP3dBJetTags
akPu5PFTrackCountingHighEffBJetTags = akPu5PFbTagger.TrackCountingHighEffBJetTags
akPu5PFTrackCountingHighPurBJetTags = akPu5PFbTagger.TrackCountingHighPurBJetTags
akPu5PFPatJetPartonAssociationLegacy = akPu5PFbTagger.PatJetPartonAssociationLegacy

akPu5PFImpactParameterTagInfos = akPu5PFbTagger.ImpactParameterTagInfos
akPu5PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu5PFJetProbabilityBJetTags = akPu5PFbTagger.JetProbabilityBJetTags
akPu5PFPositiveOnlyJetProbabilityBJetTags = akPu5PFbTagger.PositiveOnlyJetProbabilityBJetTags
akPu5PFNegativeOnlyJetProbabilityBJetTags = akPu5PFbTagger.NegativeOnlyJetProbabilityBJetTags
akPu5PFNegativeTrackCountingHighEffBJetTags = akPu5PFbTagger.NegativeTrackCountingHighEffBJetTags
akPu5PFNegativeTrackCountingHighPurBJetTags = akPu5PFbTagger.NegativeTrackCountingHighPurBJetTags
akPu5PFNegativeOnlyJetBProbabilityBJetTags = akPu5PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akPu5PFPositiveOnlyJetBProbabilityBJetTags = akPu5PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akPu5PFSecondaryVertexTagInfos = akPu5PFbTagger.SecondaryVertexTagInfos
akPu5PFSimpleSecondaryVertexHighEffBJetTags = akPu5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akPu5PFSimpleSecondaryVertexHighPurBJetTags = akPu5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akPu5PFCombinedSecondaryVertexBJetTags = akPu5PFbTagger.CombinedSecondaryVertexBJetTags
akPu5PFCombinedSecondaryVertexV2BJetTags = akPu5PFbTagger.CombinedSecondaryVertexV2BJetTags

akPu5PFSecondaryVertexNegativeTagInfos = akPu5PFbTagger.SecondaryVertexNegativeTagInfos
akPu5PFNegativeSimpleSecondaryVertexHighEffBJetTags = akPu5PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akPu5PFNegativeSimpleSecondaryVertexHighPurBJetTags = akPu5PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akPu5PFNegativeCombinedSecondaryVertexBJetTags = akPu5PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akPu5PFPositiveCombinedSecondaryVertexBJetTags = akPu5PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akPu5PFSoftPFMuonsTagInfos = akPu5PFbTagger.SoftPFMuonsTagInfos
akPu5PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akPu5PFSoftPFMuonBJetTags = akPu5PFbTagger.SoftPFMuonBJetTags
akPu5PFSoftPFMuonByIP3dBJetTags = akPu5PFbTagger.SoftPFMuonByIP3dBJetTags
akPu5PFSoftPFMuonByPtBJetTags = akPu5PFbTagger.SoftPFMuonByPtBJetTags
akPu5PFNegativeSoftPFMuonByPtBJetTags = akPu5PFbTagger.NegativeSoftPFMuonByPtBJetTags
akPu5PFPositiveSoftPFMuonByPtBJetTags = akPu5PFbTagger.PositiveSoftPFMuonByPtBJetTags
akPu5PFPatJetFlavourIdLegacy = cms.Sequence(akPu5PFPatJetPartonAssociationLegacy*akPu5PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akPu5PFPatJetFlavourAssociation = akPu5PFbTagger.PatJetFlavourAssociation
#akPu5PFPatJetFlavourId = cms.Sequence(akPu5PFPatJetPartons*akPu5PFPatJetFlavourAssociation)

akPu5PFJetBtaggingIP       = cms.Sequence(akPu5PFImpactParameterTagInfos *
            (akPu5PFTrackCountingHighEffBJetTags +
             akPu5PFTrackCountingHighPurBJetTags +
             akPu5PFJetProbabilityBJetTags +
             akPu5PFJetBProbabilityBJetTags +
             akPu5PFPositiveOnlyJetProbabilityBJetTags +
             akPu5PFNegativeOnlyJetProbabilityBJetTags +
             akPu5PFNegativeTrackCountingHighEffBJetTags +
             akPu5PFNegativeTrackCountingHighPurBJetTags +
             akPu5PFNegativeOnlyJetBProbabilityBJetTags +
             akPu5PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akPu5PFJetBtaggingSV = cms.Sequence(akPu5PFImpactParameterTagInfos
            *
            akPu5PFSecondaryVertexTagInfos
            * (akPu5PFSimpleSecondaryVertexHighEffBJetTags
                +
                akPu5PFSimpleSecondaryVertexHighPurBJetTags
                +
                akPu5PFCombinedSecondaryVertexBJetTags
                +
                akPu5PFCombinedSecondaryVertexV2BJetTags
              )
            )

akPu5PFJetBtaggingNegSV = cms.Sequence(akPu5PFImpactParameterTagInfos
            *
            akPu5PFSecondaryVertexNegativeTagInfos
            * (akPu5PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akPu5PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akPu5PFNegativeCombinedSecondaryVertexBJetTags
                +
                akPu5PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akPu5PFJetBtaggingMu = cms.Sequence(akPu5PFSoftPFMuonsTagInfos * (akPu5PFSoftPFMuonBJetTags
                +
                akPu5PFSoftPFMuonByIP3dBJetTags
                +
                akPu5PFSoftPFMuonByPtBJetTags
                +
                akPu5PFNegativeSoftPFMuonByPtBJetTags
                +
                akPu5PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akPu5PFJetBtagging = cms.Sequence(akPu5PFJetBtaggingIP
            *akPu5PFJetBtaggingSV
            *akPu5PFJetBtaggingNegSV
#            *akPu5PFJetBtaggingMu
            )

akPu5PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akPu5PFJets"),
        genJetMatch          = cms.InputTag("akPu5PFmatch"),
        genPartonMatch       = cms.InputTag("akPu5PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akPu5PFcorr")),
        JetPartonMapSource   = cms.InputTag("akPu5PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akPu5PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akPu5PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akPu5PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akPu5PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akPu5PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akPu5PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akPu5PFJetBProbabilityBJetTags"),
            cms.InputTag("akPu5PFJetProbabilityBJetTags"),
            #cms.InputTag("akPu5PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akPu5PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akPu5PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akPu5PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akPu5PFJetID"),
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

akPu5PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akPu5PFJets"),
           	    R0  = cms.double( 0.5)
)
akPu5PFpatJetsWithBtagging.userData.userFloats.src += ['akPu5PFNjettiness:tau1','akPu5PFNjettiness:tau2','akPu5PFNjettiness:tau3']

akPu5PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akPu5PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akPu5PF"),
                                                             jetName = cms.untracked.string("akPu5PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akPu5PFJetSequence_mc = cms.Sequence(
                                                  #akPu5PFclean
                                                  #*
                                                  akPu5PFmatch
                                                  *
                                                  akPu5PFparton
                                                  *
                                                  akPu5PFcorr
                                                  *
                                                  #akPu5PFJetID
                                                  #*
                                                  akPu5PFPatJetFlavourIdLegacy
                                                  #*
			                          #akPu5PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akPu5PFJetTracksAssociatorAtVertex
                                                  *
                                                  akPu5PFJetBtagging
                                                  *
                                                  akPu5PFNjettiness
                                                  *
                                                  akPu5PFpatJetsWithBtagging
                                                  *
                                                  akPu5PFJetAnalyzer
                                                  )

akPu5PFJetSequence_data = cms.Sequence(akPu5PFcorr
                                                    *
                                                    #akPu5PFJetID
                                                    #*
                                                    akPu5PFJetTracksAssociatorAtVertex
                                                    *
                                                    akPu5PFJetBtagging
                                                    *
                                                    akPu5PFNjettiness 
                                                    *
                                                    akPu5PFpatJetsWithBtagging
                                                    *
                                                    akPu5PFJetAnalyzer
                                                    )

akPu5PFJetSequence_jec = cms.Sequence(akPu5PFJetSequence_mc)
akPu5PFJetSequence_mix = cms.Sequence(akPu5PFJetSequence_mc)

akPu5PFJetSequence = cms.Sequence(akPu5PFJetSequence_data)
