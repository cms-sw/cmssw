

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCs3PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCs3PFJets"),
    matched = cms.InputTag("ak3HiSignalGenJets"),
    resolveByMatchQuality = cms.bool(True),
    maxDeltaR = 0.3
    )

akCs3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs3PFJets")
                                                        )

akCs3PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCs3PFJets"),
    payload = "AK3PF_offline"
    )

akCs3PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCs3CaloJets'))

#akCs3PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiSignalGenJets'))

akCs3PFbTagger = bTaggers("akCs3PF",0.3)

#create objects locally since they dont load properly otherwise
#akCs3PFmatch = akCs3PFbTagger.match
akCs3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCs3PFJets"), matched = cms.InputTag("hiSignalGenParticles"))
akCs3PFPatJetFlavourAssociationLegacy = akCs3PFbTagger.PatJetFlavourAssociationLegacy
akCs3PFPatJetPartons = akCs3PFbTagger.PatJetPartons
akCs3PFJetTracksAssociatorAtVertex = akCs3PFbTagger.JetTracksAssociatorAtVertex
akCs3PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCs3PFSimpleSecondaryVertexHighEffBJetTags = akCs3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs3PFSimpleSecondaryVertexHighPurBJetTags = akCs3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs3PFCombinedSecondaryVertexBJetTags = akCs3PFbTagger.CombinedSecondaryVertexBJetTags
akCs3PFCombinedSecondaryVertexV2BJetTags = akCs3PFbTagger.CombinedSecondaryVertexV2BJetTags
akCs3PFJetBProbabilityBJetTags = akCs3PFbTagger.JetBProbabilityBJetTags
akCs3PFSoftPFMuonByPtBJetTags = akCs3PFbTagger.SoftPFMuonByPtBJetTags
akCs3PFSoftPFMuonByIP3dBJetTags = akCs3PFbTagger.SoftPFMuonByIP3dBJetTags
akCs3PFTrackCountingHighEffBJetTags = akCs3PFbTagger.TrackCountingHighEffBJetTags
akCs3PFTrackCountingHighPurBJetTags = akCs3PFbTagger.TrackCountingHighPurBJetTags
akCs3PFPatJetPartonAssociationLegacy = akCs3PFbTagger.PatJetPartonAssociationLegacy

akCs3PFImpactParameterTagInfos = akCs3PFbTagger.ImpactParameterTagInfos
akCs3PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs3PFJetProbabilityBJetTags = akCs3PFbTagger.JetProbabilityBJetTags

akCs3PFSecondaryVertexTagInfos = akCs3PFbTagger.SecondaryVertexTagInfos
akCs3PFSimpleSecondaryVertexHighEffBJetTags = akCs3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCs3PFSimpleSecondaryVertexHighPurBJetTags = akCs3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCs3PFCombinedSecondaryVertexBJetTags = akCs3PFbTagger.CombinedSecondaryVertexBJetTags
akCs3PFCombinedSecondaryVertexV2BJetTags = akCs3PFbTagger.CombinedSecondaryVertexV2BJetTags

akCs3PFSecondaryVertexNegativeTagInfos = akCs3PFbTagger.SecondaryVertexNegativeTagInfos
akCs3PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCs3PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCs3PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCs3PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCs3PFNegativeCombinedSecondaryVertexBJetTags = akCs3PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCs3PFPositiveCombinedSecondaryVertexBJetTags = akCs3PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCs3PFNegativeCombinedSecondaryVertexV2BJetTags = akCs3PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCs3PFPositiveCombinedSecondaryVertexV2BJetTags = akCs3PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCs3PFSoftPFMuonsTagInfos = akCs3PFbTagger.SoftPFMuonsTagInfos
akCs3PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCs3PFSoftPFMuonBJetTags = akCs3PFbTagger.SoftPFMuonBJetTags
akCs3PFSoftPFMuonByIP3dBJetTags = akCs3PFbTagger.SoftPFMuonByIP3dBJetTags
akCs3PFSoftPFMuonByPtBJetTags = akCs3PFbTagger.SoftPFMuonByPtBJetTags
akCs3PFNegativeSoftPFMuonByPtBJetTags = akCs3PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCs3PFPositiveSoftPFMuonByPtBJetTags = akCs3PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCs3PFPatJetFlavourIdLegacy = cms.Sequence(akCs3PFPatJetPartonAssociationLegacy*akCs3PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCs3PFPatJetFlavourAssociation = akCs3PFbTagger.PatJetFlavourAssociation
#akCs3PFPatJetFlavourId = cms.Sequence(akCs3PFPatJetPartons*akCs3PFPatJetFlavourAssociation)

akCs3PFJetBtaggingIP       = cms.Sequence(akCs3PFImpactParameterTagInfos *
            (akCs3PFTrackCountingHighEffBJetTags +
             akCs3PFTrackCountingHighPurBJetTags +
             akCs3PFJetProbabilityBJetTags +
             akCs3PFJetBProbabilityBJetTags 
            )
            )

akCs3PFJetBtaggingSV = cms.Sequence(akCs3PFImpactParameterTagInfos
            *
            akCs3PFSecondaryVertexTagInfos
            * (akCs3PFSimpleSecondaryVertexHighEffBJetTags+
                akCs3PFSimpleSecondaryVertexHighPurBJetTags+
                akCs3PFCombinedSecondaryVertexBJetTags+
                akCs3PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCs3PFJetBtaggingNegSV = cms.Sequence(akCs3PFImpactParameterTagInfos
            *
            akCs3PFSecondaryVertexNegativeTagInfos
            * (akCs3PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCs3PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCs3PFNegativeCombinedSecondaryVertexBJetTags+
                akCs3PFPositiveCombinedSecondaryVertexBJetTags+
                akCs3PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCs3PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCs3PFJetBtaggingMu = cms.Sequence(akCs3PFSoftPFMuonsTagInfos * (akCs3PFSoftPFMuonBJetTags
                +
                akCs3PFSoftPFMuonByIP3dBJetTags
                +
                akCs3PFSoftPFMuonByPtBJetTags
                +
                akCs3PFNegativeSoftPFMuonByPtBJetTags
                +
                akCs3PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCs3PFJetBtagging = cms.Sequence(akCs3PFJetBtaggingIP
            *akCs3PFJetBtaggingSV
            *akCs3PFJetBtaggingNegSV
#            *akCs3PFJetBtaggingMu
            )

akCs3PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCs3PFJets"),
        genJetMatch          = cms.InputTag("akCs3PFmatch"),
        genPartonMatch       = cms.InputTag("akCs3PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCs3PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCs3PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCs3PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCs3PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCs3PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCs3PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCs3PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCs3PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCs3PFJetBProbabilityBJetTags"),
            cms.InputTag("akCs3PFJetProbabilityBJetTags"),
            #cms.InputTag("akCs3PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCs3PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCs3PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCs3PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCs3PFJetID"),
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

akCs3PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCs3PFJets"),
           	    R0  = cms.double( 0.3)
)
akCs3PFpatJetsWithBtagging.userData.userFloats.src += ['akCs3PFNjettiness:tau1','akCs3PFNjettiness:tau2','akCs3PFNjettiness:tau3']

akCs3PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCs3PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak3HiGenJets',
                                                             rParam = 0.3,
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
                                                             bTagJetName = cms.untracked.string("akCs3PF"),
                                                             jetName = cms.untracked.string("akCs3PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(False)
                                                             )

akCs3PFJetSequence_mc = cms.Sequence(
                                                  #akCs3PFclean
                                                  #*
                                                  akCs3PFmatch
                                                  *
                                                  akCs3PFparton
                                                  *
                                                  akCs3PFcorr
                                                  *
                                                  #akCs3PFJetID
                                                  #*
                                                  akCs3PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCs3PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCs3PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCs3PFJetBtagging
                                                  *
                                                  akCs3PFNjettiness
                                                  *
                                                  akCs3PFpatJetsWithBtagging
                                                  *
                                                  akCs3PFJetAnalyzer
                                                  )

akCs3PFJetSequence_data = cms.Sequence(akCs3PFcorr
                                                    *
                                                    #akCs3PFJetID
                                                    #*
                                                    akCs3PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCs3PFJetBtagging
                                                    *
                                                    akCs3PFNjettiness 
                                                    *
                                                    akCs3PFpatJetsWithBtagging
                                                    *
                                                    akCs3PFJetAnalyzer
                                                    )

akCs3PFJetSequence_jec = cms.Sequence(akCs3PFJetSequence_mc)
akCs3PFJetSequence_mb = cms.Sequence(akCs3PFJetSequence_mc)

akCs3PFJetSequence = cms.Sequence(akCs3PFJetSequence_jec)
akCs3PFJetAnalyzer.genPtMin = cms.untracked.double(1)
akCs3PFJetAnalyzer.jetPtMin = cms.double(1)
