

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter1PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter1PFJets"),
    matched = cms.InputTag("ak1HiSignalGenJets"),
    maxDeltaR = 0.1
    )

akCsFilter1PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter1PFJets")
                                                        )

akCsFilter1PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter1PFJets"),
    payload = "AK1PF_offline"
    )

akCsFilter1PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter1CaloJets'))

#akCsFilter1PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak1HiSignalGenJets'))

akCsFilter1PFbTagger = bTaggers("akCsFilter1PF",0.1)

#create objects locally since they dont load properly otherwise
#akCsFilter1PFmatch = akCsFilter1PFbTagger.match
akCsFilter1PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter1PFJets"), matched = cms.InputTag("selectedPartons"))
akCsFilter1PFPatJetFlavourAssociationLegacy = akCsFilter1PFbTagger.PatJetFlavourAssociationLegacy
akCsFilter1PFPatJetPartons = akCsFilter1PFbTagger.PatJetPartons
akCsFilter1PFJetTracksAssociatorAtVertex = akCsFilter1PFbTagger.JetTracksAssociatorAtVertex
akCsFilter1PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter1PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter1PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter1PFCombinedSecondaryVertexBJetTags = akCsFilter1PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter1PFCombinedSecondaryVertexV2BJetTags = akCsFilter1PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter1PFJetBProbabilityBJetTags = akCsFilter1PFbTagger.JetBProbabilityBJetTags
akCsFilter1PFSoftPFMuonByPtBJetTags = akCsFilter1PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter1PFSoftPFMuonByIP3dBJetTags = akCsFilter1PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter1PFTrackCountingHighEffBJetTags = akCsFilter1PFbTagger.TrackCountingHighEffBJetTags
akCsFilter1PFTrackCountingHighPurBJetTags = akCsFilter1PFbTagger.TrackCountingHighPurBJetTags
akCsFilter1PFPatJetPartonAssociationLegacy = akCsFilter1PFbTagger.PatJetPartonAssociationLegacy

akCsFilter1PFImpactParameterTagInfos = akCsFilter1PFbTagger.ImpactParameterTagInfos
akCsFilter1PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter1PFJetProbabilityBJetTags = akCsFilter1PFbTagger.JetProbabilityBJetTags

akCsFilter1PFSecondaryVertexTagInfos = akCsFilter1PFbTagger.SecondaryVertexTagInfos
akCsFilter1PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter1PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter1PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter1PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter1PFCombinedSecondaryVertexBJetTags = akCsFilter1PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter1PFCombinedSecondaryVertexV2BJetTags = akCsFilter1PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter1PFSecondaryVertexNegativeTagInfos = akCsFilter1PFbTagger.SecondaryVertexNegativeTagInfos
akCsFilter1PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter1PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter1PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter1PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter1PFNegativeCombinedSecondaryVertexBJetTags = akCsFilter1PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter1PFPositiveCombinedSecondaryVertexBJetTags = akCsFilter1PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter1PFNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter1PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter1PFPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter1PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter1PFSoftPFMuonsTagInfos = akCsFilter1PFbTagger.SoftPFMuonsTagInfos
akCsFilter1PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter1PFSoftPFMuonBJetTags = akCsFilter1PFbTagger.SoftPFMuonBJetTags
akCsFilter1PFSoftPFMuonByIP3dBJetTags = akCsFilter1PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter1PFSoftPFMuonByPtBJetTags = akCsFilter1PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter1PFNegativeSoftPFMuonByPtBJetTags = akCsFilter1PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter1PFPositiveSoftPFMuonByPtBJetTags = akCsFilter1PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter1PFPatJetFlavourIdLegacy = cms.Sequence(akCsFilter1PFPatJetPartonAssociationLegacy*akCsFilter1PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter1PFPatJetFlavourAssociation = akCsFilter1PFbTagger.PatJetFlavourAssociation
#akCsFilter1PFPatJetFlavourId = cms.Sequence(akCsFilter1PFPatJetPartons*akCsFilter1PFPatJetFlavourAssociation)

akCsFilter1PFJetBtaggingIP       = cms.Sequence(akCsFilter1PFImpactParameterTagInfos *
            (akCsFilter1PFTrackCountingHighEffBJetTags +
             akCsFilter1PFTrackCountingHighPurBJetTags +
             akCsFilter1PFJetProbabilityBJetTags +
             akCsFilter1PFJetBProbabilityBJetTags 
            )
            )

akCsFilter1PFJetBtaggingSV = cms.Sequence(akCsFilter1PFImpactParameterTagInfos
            *
            akCsFilter1PFSecondaryVertexTagInfos
            * (akCsFilter1PFSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter1PFSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter1PFCombinedSecondaryVertexBJetTags+
                akCsFilter1PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter1PFJetBtaggingNegSV = cms.Sequence(akCsFilter1PFImpactParameterTagInfos
            *
            akCsFilter1PFSecondaryVertexNegativeTagInfos
            * (akCsFilter1PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter1PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter1PFNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter1PFPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter1PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter1PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter1PFJetBtaggingMu = cms.Sequence(akCsFilter1PFSoftPFMuonsTagInfos * (akCsFilter1PFSoftPFMuonBJetTags
                +
                akCsFilter1PFSoftPFMuonByIP3dBJetTags
                +
                akCsFilter1PFSoftPFMuonByPtBJetTags
                +
                akCsFilter1PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter1PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter1PFJetBtagging = cms.Sequence(akCsFilter1PFJetBtaggingIP
            *akCsFilter1PFJetBtaggingSV
            *akCsFilter1PFJetBtaggingNegSV
#            *akCsFilter1PFJetBtaggingMu
            )

akCsFilter1PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter1PFJets"),
        genJetMatch          = cms.InputTag("akCsFilter1PFmatch"),
        genPartonMatch       = cms.InputTag("akCsFilter1PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter1PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter1PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter1PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter1PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter1PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter1PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter1PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter1PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter1PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter1PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter1PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter1PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter1PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter1PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter1PFJetID"),
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

akCsFilter1PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter1PFJets"),
           	    R0  = cms.double( 0.1)
)
akCsFilter1PFpatJetsWithBtagging.userData.userFloats.src += ['akCsFilter1PFNjettiness:tau1','akCsFilter1PFNjettiness:tau2','akCsFilter1PFNjettiness:tau3']

akCsFilter1PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter1PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak1HiGenJets',
                                                             rParam = 0.1,
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
                                                             bTagJetName = cms.untracked.string("akCsFilter1PF"),
                                                             jetName = cms.untracked.string("akCsFilter1PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter1PFJetSequence_mc = cms.Sequence(
                                                  #akCsFilter1PFclean
                                                  #*
                                                  akCsFilter1PFmatch
                                                  *
                                                  akCsFilter1PFparton
                                                  *
                                                  akCsFilter1PFcorr
                                                  *
                                                  #akCsFilter1PFJetID
                                                  #*
                                                  akCsFilter1PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter1PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter1PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter1PFJetBtagging
                                                  *
                                                  akCsFilter1PFNjettiness
                                                  *
                                                  akCsFilter1PFpatJetsWithBtagging
                                                  *
                                                  akCsFilter1PFJetAnalyzer
                                                  )

akCsFilter1PFJetSequence_data = cms.Sequence(akCsFilter1PFcorr
                                                    *
                                                    #akCsFilter1PFJetID
                                                    #*
                                                    akCsFilter1PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter1PFJetBtagging
                                                    *
                                                    akCsFilter1PFNjettiness 
                                                    *
                                                    akCsFilter1PFpatJetsWithBtagging
                                                    *
                                                    akCsFilter1PFJetAnalyzer
                                                    )

akCsFilter1PFJetSequence_jec = cms.Sequence(akCsFilter1PFJetSequence_mc)
akCsFilter1PFJetSequence_mb = cms.Sequence(akCsFilter1PFJetSequence_mc)

akCsFilter1PFJetSequence = cms.Sequence(akCsFilter1PFJetSequence_mc)
