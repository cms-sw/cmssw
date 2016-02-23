

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akVs5PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akVs5PFJets"),
    matched = cms.InputTag("ak5HiGenJets"),
    maxDeltaR = 0.5
    )

akVs5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs5PFJets")
                                                        )

akVs5PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akVs5PFJets"),
    payload = "AK5PF_offline"
    )

akVs5PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akVs5CaloJets'))

#akVs5PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiGenJets'))

akVs5PFbTagger = bTaggers("akVs5PF",0.5)

#create objects locally since they dont load properly otherwise
#akVs5PFmatch = akVs5PFbTagger.match
akVs5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akVs5PFJets"), matched = cms.InputTag("genParticles"))
akVs5PFPatJetFlavourAssociationLegacy = akVs5PFbTagger.PatJetFlavourAssociationLegacy
akVs5PFPatJetPartons = akVs5PFbTagger.PatJetPartons
akVs5PFJetTracksAssociatorAtVertex = akVs5PFbTagger.JetTracksAssociatorAtVertex
akVs5PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akVs5PFSimpleSecondaryVertexHighEffBJetTags = akVs5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs5PFSimpleSecondaryVertexHighPurBJetTags = akVs5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs5PFCombinedSecondaryVertexBJetTags = akVs5PFbTagger.CombinedSecondaryVertexBJetTags
akVs5PFCombinedSecondaryVertexV2BJetTags = akVs5PFbTagger.CombinedSecondaryVertexV2BJetTags
akVs5PFJetBProbabilityBJetTags = akVs5PFbTagger.JetBProbabilityBJetTags
akVs5PFSoftPFMuonByPtBJetTags = akVs5PFbTagger.SoftPFMuonByPtBJetTags
akVs5PFSoftPFMuonByIP3dBJetTags = akVs5PFbTagger.SoftPFMuonByIP3dBJetTags
akVs5PFTrackCountingHighEffBJetTags = akVs5PFbTagger.TrackCountingHighEffBJetTags
akVs5PFTrackCountingHighPurBJetTags = akVs5PFbTagger.TrackCountingHighPurBJetTags
akVs5PFPatJetPartonAssociationLegacy = akVs5PFbTagger.PatJetPartonAssociationLegacy

akVs5PFImpactParameterTagInfos = akVs5PFbTagger.ImpactParameterTagInfos
akVs5PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs5PFJetProbabilityBJetTags = akVs5PFbTagger.JetProbabilityBJetTags
akVs5PFPositiveOnlyJetProbabilityBJetTags = akVs5PFbTagger.PositiveOnlyJetProbabilityBJetTags
akVs5PFNegativeOnlyJetProbabilityBJetTags = akVs5PFbTagger.NegativeOnlyJetProbabilityBJetTags
akVs5PFNegativeTrackCountingHighEffBJetTags = akVs5PFbTagger.NegativeTrackCountingHighEffBJetTags
akVs5PFNegativeTrackCountingHighPurBJetTags = akVs5PFbTagger.NegativeTrackCountingHighPurBJetTags
akVs5PFNegativeOnlyJetBProbabilityBJetTags = akVs5PFbTagger.NegativeOnlyJetBProbabilityBJetTags
akVs5PFPositiveOnlyJetBProbabilityBJetTags = akVs5PFbTagger.PositiveOnlyJetBProbabilityBJetTags

akVs5PFSecondaryVertexTagInfos = akVs5PFbTagger.SecondaryVertexTagInfos
akVs5PFSimpleSecondaryVertexHighEffBJetTags = akVs5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akVs5PFSimpleSecondaryVertexHighPurBJetTags = akVs5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akVs5PFCombinedSecondaryVertexBJetTags = akVs5PFbTagger.CombinedSecondaryVertexBJetTags
akVs5PFCombinedSecondaryVertexV2BJetTags = akVs5PFbTagger.CombinedSecondaryVertexV2BJetTags

akVs5PFSecondaryVertexNegativeTagInfos = akVs5PFbTagger.SecondaryVertexNegativeTagInfos
akVs5PFNegativeSimpleSecondaryVertexHighEffBJetTags = akVs5PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akVs5PFNegativeSimpleSecondaryVertexHighPurBJetTags = akVs5PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akVs5PFNegativeCombinedSecondaryVertexBJetTags = akVs5PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akVs5PFPositiveCombinedSecondaryVertexBJetTags = akVs5PFbTagger.PositiveCombinedSecondaryVertexBJetTags

akVs5PFSoftPFMuonsTagInfos = akVs5PFbTagger.SoftPFMuonsTagInfos
akVs5PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akVs5PFSoftPFMuonBJetTags = akVs5PFbTagger.SoftPFMuonBJetTags
akVs5PFSoftPFMuonByIP3dBJetTags = akVs5PFbTagger.SoftPFMuonByIP3dBJetTags
akVs5PFSoftPFMuonByPtBJetTags = akVs5PFbTagger.SoftPFMuonByPtBJetTags
akVs5PFNegativeSoftPFMuonByPtBJetTags = akVs5PFbTagger.NegativeSoftPFMuonByPtBJetTags
akVs5PFPositiveSoftPFMuonByPtBJetTags = akVs5PFbTagger.PositiveSoftPFMuonByPtBJetTags
akVs5PFPatJetFlavourIdLegacy = cms.Sequence(akVs5PFPatJetPartonAssociationLegacy*akVs5PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akVs5PFPatJetFlavourAssociation = akVs5PFbTagger.PatJetFlavourAssociation
#akVs5PFPatJetFlavourId = cms.Sequence(akVs5PFPatJetPartons*akVs5PFPatJetFlavourAssociation)

akVs5PFJetBtaggingIP       = cms.Sequence(akVs5PFImpactParameterTagInfos *
            (akVs5PFTrackCountingHighEffBJetTags +
             akVs5PFTrackCountingHighPurBJetTags +
             akVs5PFJetProbabilityBJetTags +
             akVs5PFJetBProbabilityBJetTags +
             akVs5PFPositiveOnlyJetProbabilityBJetTags +
             akVs5PFNegativeOnlyJetProbabilityBJetTags +
             akVs5PFNegativeTrackCountingHighEffBJetTags +
             akVs5PFNegativeTrackCountingHighPurBJetTags +
             akVs5PFNegativeOnlyJetBProbabilityBJetTags +
             akVs5PFPositiveOnlyJetBProbabilityBJetTags
            )
            )

akVs5PFJetBtaggingSV = cms.Sequence(akVs5PFImpactParameterTagInfos
            *
            akVs5PFSecondaryVertexTagInfos
            * (akVs5PFSimpleSecondaryVertexHighEffBJetTags
                +
                akVs5PFSimpleSecondaryVertexHighPurBJetTags
                +
                akVs5PFCombinedSecondaryVertexBJetTags
                +
                akVs5PFCombinedSecondaryVertexV2BJetTags
              )
            )

akVs5PFJetBtaggingNegSV = cms.Sequence(akVs5PFImpactParameterTagInfos
            *
            akVs5PFSecondaryVertexNegativeTagInfos
            * (akVs5PFNegativeSimpleSecondaryVertexHighEffBJetTags
                +
                akVs5PFNegativeSimpleSecondaryVertexHighPurBJetTags
                +
                akVs5PFNegativeCombinedSecondaryVertexBJetTags
                +
                akVs5PFPositiveCombinedSecondaryVertexBJetTags
              )
            )

akVs5PFJetBtaggingMu = cms.Sequence(akVs5PFSoftPFMuonsTagInfos * (akVs5PFSoftPFMuonBJetTags
                +
                akVs5PFSoftPFMuonByIP3dBJetTags
                +
                akVs5PFSoftPFMuonByPtBJetTags
                +
                akVs5PFNegativeSoftPFMuonByPtBJetTags
                +
                akVs5PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akVs5PFJetBtagging = cms.Sequence(akVs5PFJetBtaggingIP
            *akVs5PFJetBtaggingSV
            *akVs5PFJetBtaggingNegSV
#            *akVs5PFJetBtaggingMu
            )

akVs5PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akVs5PFJets"),
        genJetMatch          = cms.InputTag("akVs5PFmatch"),
        genPartonMatch       = cms.InputTag("akVs5PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akVs5PFcorr")),
        JetPartonMapSource   = cms.InputTag("akVs5PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akVs5PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akVs5PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akVs5PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akVs5PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akVs5PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akVs5PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akVs5PFJetBProbabilityBJetTags"),
            cms.InputTag("akVs5PFJetProbabilityBJetTags"),
            #cms.InputTag("akVs5PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akVs5PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akVs5PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akVs5PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akVs5PFJetID"),
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

akVs5PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akVs5PFJets"),
           	    R0  = cms.double( 0.5)
)
akVs5PFpatJetsWithBtagging.userData.userFloats.src += ['akVs5PFNjettiness:tau1','akVs5PFNjettiness:tau2','akVs5PFNjettiness:tau3']

akVs5PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akVs5PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak5HiGenJets',
                                                             rParam = 0.5,
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
                                                             bTagJetName = cms.untracked.string("akVs5PF"),
                                                             jetName = cms.untracked.string("akVs5PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True)
                                                             )

akVs5PFJetSequence_mc = cms.Sequence(
                                                  #akVs5PFclean
                                                  #*
                                                  akVs5PFmatch
                                                  *
                                                  akVs5PFparton
                                                  *
                                                  akVs5PFcorr
                                                  *
                                                  #akVs5PFJetID
                                                  #*
                                                  akVs5PFPatJetFlavourIdLegacy
                                                  #*
			                          #akVs5PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akVs5PFJetTracksAssociatorAtVertex
                                                  *
                                                  akVs5PFJetBtagging
                                                  *
                                                  akVs5PFNjettiness
                                                  *
                                                  akVs5PFpatJetsWithBtagging
                                                  *
                                                  akVs5PFJetAnalyzer
                                                  )

akVs5PFJetSequence_data = cms.Sequence(akVs5PFcorr
                                                    *
                                                    #akVs5PFJetID
                                                    #*
                                                    akVs5PFJetTracksAssociatorAtVertex
                                                    *
                                                    akVs5PFJetBtagging
                                                    *
                                                    akVs5PFNjettiness 
                                                    *
                                                    akVs5PFpatJetsWithBtagging
                                                    *
                                                    akVs5PFJetAnalyzer
                                                    )

akVs5PFJetSequence_jec = cms.Sequence(akVs5PFJetSequence_mc)
akVs5PFJetSequence_mix = cms.Sequence(akVs5PFJetSequence_mc)

akVs5PFJetSequence = cms.Sequence(akVs5PFJetSequence_mc)
