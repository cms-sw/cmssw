

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter2PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter2PFJets"),
    matched = cms.InputTag("ak2HiSignalGenJets"),
    resolveByMatchQuality = cms.bool(True),
    maxDeltaR = 0.2
    )

akCsFilter2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter2PFJets")
                                                        )

akCsFilter2PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter2PFJets"),
    payload = "AK2PF_offline"
    )

akCsFilter2PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter2CaloJets'))

#akCsFilter2PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak2HiSignalGenJets'))

akCsFilter2PFbTagger = bTaggers("akCsFilter2PF",0.2)

#create objects locally since they dont load properly otherwise
#akCsFilter2PFmatch = akCsFilter2PFbTagger.match
akCsFilter2PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter2PFJets"), matched = cms.InputTag("hiSignalGenParticles"))
akCsFilter2PFPatJetFlavourAssociationLegacy = akCsFilter2PFbTagger.PatJetFlavourAssociationLegacy
akCsFilter2PFPatJetPartons = akCsFilter2PFbTagger.PatJetPartons
akCsFilter2PFJetTracksAssociatorAtVertex = akCsFilter2PFbTagger.JetTracksAssociatorAtVertex
akCsFilter2PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter2PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter2PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter2PFCombinedSecondaryVertexBJetTags = akCsFilter2PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter2PFCombinedSecondaryVertexV2BJetTags = akCsFilter2PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter2PFJetBProbabilityBJetTags = akCsFilter2PFbTagger.JetBProbabilityBJetTags
akCsFilter2PFSoftPFMuonByPtBJetTags = akCsFilter2PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter2PFSoftPFMuonByIP3dBJetTags = akCsFilter2PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter2PFTrackCountingHighEffBJetTags = akCsFilter2PFbTagger.TrackCountingHighEffBJetTags
akCsFilter2PFTrackCountingHighPurBJetTags = akCsFilter2PFbTagger.TrackCountingHighPurBJetTags
akCsFilter2PFPatJetPartonAssociationLegacy = akCsFilter2PFbTagger.PatJetPartonAssociationLegacy

akCsFilter2PFImpactParameterTagInfos = akCsFilter2PFbTagger.ImpactParameterTagInfos
akCsFilter2PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter2PFJetProbabilityBJetTags = akCsFilter2PFbTagger.JetProbabilityBJetTags

akCsFilter2PFSecondaryVertexTagInfos = akCsFilter2PFbTagger.SecondaryVertexTagInfos
akCsFilter2PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter2PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter2PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter2PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter2PFCombinedSecondaryVertexBJetTags = akCsFilter2PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter2PFCombinedSecondaryVertexV2BJetTags = akCsFilter2PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter2PFSecondaryVertexNegativeTagInfos = akCsFilter2PFbTagger.SecondaryVertexNegativeTagInfos
akCsFilter2PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter2PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter2PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter2PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter2PFNegativeCombinedSecondaryVertexBJetTags = akCsFilter2PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter2PFPositiveCombinedSecondaryVertexBJetTags = akCsFilter2PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter2PFNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter2PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter2PFPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter2PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter2PFSoftPFMuonsTagInfos = akCsFilter2PFbTagger.SoftPFMuonsTagInfos
akCsFilter2PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter2PFSoftPFMuonBJetTags = akCsFilter2PFbTagger.SoftPFMuonBJetTags
akCsFilter2PFSoftPFMuonByIP3dBJetTags = akCsFilter2PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter2PFSoftPFMuonByPtBJetTags = akCsFilter2PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter2PFNegativeSoftPFMuonByPtBJetTags = akCsFilter2PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter2PFPositiveSoftPFMuonByPtBJetTags = akCsFilter2PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter2PFPatJetFlavourIdLegacy = cms.Sequence(akCsFilter2PFPatJetPartonAssociationLegacy*akCsFilter2PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter2PFPatJetFlavourAssociation = akCsFilter2PFbTagger.PatJetFlavourAssociation
#akCsFilter2PFPatJetFlavourId = cms.Sequence(akCsFilter2PFPatJetPartons*akCsFilter2PFPatJetFlavourAssociation)

akCsFilter2PFJetBtaggingIP       = cms.Sequence(akCsFilter2PFImpactParameterTagInfos *
            (akCsFilter2PFTrackCountingHighEffBJetTags +
             akCsFilter2PFTrackCountingHighPurBJetTags +
             akCsFilter2PFJetProbabilityBJetTags +
             akCsFilter2PFJetBProbabilityBJetTags 
            )
            )

akCsFilter2PFJetBtaggingSV = cms.Sequence(akCsFilter2PFImpactParameterTagInfos
            *
            akCsFilter2PFSecondaryVertexTagInfos
            * (akCsFilter2PFSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter2PFSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter2PFCombinedSecondaryVertexBJetTags+
                akCsFilter2PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter2PFJetBtaggingNegSV = cms.Sequence(akCsFilter2PFImpactParameterTagInfos
            *
            akCsFilter2PFSecondaryVertexNegativeTagInfos
            * (akCsFilter2PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter2PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter2PFNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter2PFPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter2PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter2PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter2PFJetBtaggingMu = cms.Sequence(akCsFilter2PFSoftPFMuonsTagInfos * (akCsFilter2PFSoftPFMuonBJetTags
                +
                akCsFilter2PFSoftPFMuonByIP3dBJetTags
                +
                akCsFilter2PFSoftPFMuonByPtBJetTags
                +
                akCsFilter2PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter2PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter2PFJetBtagging = cms.Sequence(akCsFilter2PFJetBtaggingIP
            *akCsFilter2PFJetBtaggingSV
            *akCsFilter2PFJetBtaggingNegSV
#            *akCsFilter2PFJetBtaggingMu
            )

akCsFilter2PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter2PFJets"),
        genJetMatch          = cms.InputTag("akCsFilter2PFmatch"),
        genPartonMatch       = cms.InputTag("akCsFilter2PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter2PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter2PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter2PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter2PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter2PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter2PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter2PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter2PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter2PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter2PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter2PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter2PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter2PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter2PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter2PFJetID"),
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

akCsFilter2PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter2PFJets"),
           	    R0  = cms.double( 0.2)
)
akCsFilter2PFpatJetsWithBtagging.userData.userFloats.src += ['akCsFilter2PFNjettiness:tau1','akCsFilter2PFNjettiness:tau2','akCsFilter2PFNjettiness:tau3']

akCsFilter2PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter2PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsFilter2PF"),
                                                             jetName = cms.untracked.string("akCsFilter2PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter2PFJetSequence_mc = cms.Sequence(
                                                  #akCsFilter2PFclean
                                                  #*
                                                  akCsFilter2PFmatch
                                                  *
                                                  akCsFilter2PFparton
                                                  *
                                                  akCsFilter2PFcorr
                                                  *
                                                  #akCsFilter2PFJetID
                                                  #*
                                                  akCsFilter2PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter2PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter2PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter2PFJetBtagging
                                                  *
                                                  akCsFilter2PFNjettiness
                                                  *
                                                  akCsFilter2PFpatJetsWithBtagging
                                                  *
                                                  akCsFilter2PFJetAnalyzer
                                                  )

akCsFilter2PFJetSequence_data = cms.Sequence(akCsFilter2PFcorr
                                                    *
                                                    #akCsFilter2PFJetID
                                                    #*
                                                    akCsFilter2PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter2PFJetBtagging
                                                    *
                                                    akCsFilter2PFNjettiness 
                                                    *
                                                    akCsFilter2PFpatJetsWithBtagging
                                                    *
                                                    akCsFilter2PFJetAnalyzer
                                                    )

akCsFilter2PFJetSequence_jec = cms.Sequence(akCsFilter2PFJetSequence_mc)
akCsFilter2PFJetSequence_mb = cms.Sequence(akCsFilter2PFJetSequence_mc)

akCsFilter2PFJetSequence = cms.Sequence(akCsFilter2PFJetSequence_mc)
