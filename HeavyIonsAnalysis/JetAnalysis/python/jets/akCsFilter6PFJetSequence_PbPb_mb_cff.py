

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter6PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter6PFJets"),
    matched = cms.InputTag("ak6HiCleanedGenJets"),
    maxDeltaR = 0.6
    )

akCsFilter6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter6PFJets")
                                                        )

akCsFilter6PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter6PFJets"),
    payload = "AK6PF_offline"
    )

akCsFilter6PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter6CaloJets'))

#akCsFilter6PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak6HiCleanedGenJets'))

akCsFilter6PFbTagger = bTaggers("akCsFilter6PF",0.6)

#create objects locally since they dont load properly otherwise
#akCsFilter6PFmatch = akCsFilter6PFbTagger.match
akCsFilter6PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter6PFJets"), matched = cms.InputTag("selectedPartons"))
akCsFilter6PFPatJetFlavourAssociationLegacy = akCsFilter6PFbTagger.PatJetFlavourAssociationLegacy
akCsFilter6PFPatJetPartons = akCsFilter6PFbTagger.PatJetPartons
akCsFilter6PFJetTracksAssociatorAtVertex = akCsFilter6PFbTagger.JetTracksAssociatorAtVertex
akCsFilter6PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter6PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter6PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter6PFCombinedSecondaryVertexBJetTags = akCsFilter6PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter6PFCombinedSecondaryVertexV2BJetTags = akCsFilter6PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter6PFJetBProbabilityBJetTags = akCsFilter6PFbTagger.JetBProbabilityBJetTags
akCsFilter6PFSoftPFMuonByPtBJetTags = akCsFilter6PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter6PFSoftPFMuonByIP3dBJetTags = akCsFilter6PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter6PFTrackCountingHighEffBJetTags = akCsFilter6PFbTagger.TrackCountingHighEffBJetTags
akCsFilter6PFTrackCountingHighPurBJetTags = akCsFilter6PFbTagger.TrackCountingHighPurBJetTags
akCsFilter6PFPatJetPartonAssociationLegacy = akCsFilter6PFbTagger.PatJetPartonAssociationLegacy

akCsFilter6PFImpactParameterTagInfos = akCsFilter6PFbTagger.ImpactParameterTagInfos
akCsFilter6PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter6PFJetProbabilityBJetTags = akCsFilter6PFbTagger.JetProbabilityBJetTags

akCsFilter6PFSecondaryVertexTagInfos = akCsFilter6PFbTagger.SecondaryVertexTagInfos
akCsFilter6PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter6PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter6PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter6PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter6PFCombinedSecondaryVertexBJetTags = akCsFilter6PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter6PFCombinedSecondaryVertexV2BJetTags = akCsFilter6PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter6PFSecondaryVertexNegativeTagInfos = akCsFilter6PFbTagger.SecondaryVertexNegativeTagInfos
akCsFilter6PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter6PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter6PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter6PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter6PFNegativeCombinedSecondaryVertexBJetTags = akCsFilter6PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter6PFPositiveCombinedSecondaryVertexBJetTags = akCsFilter6PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter6PFNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter6PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter6PFPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter6PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter6PFSoftPFMuonsTagInfos = akCsFilter6PFbTagger.SoftPFMuonsTagInfos
akCsFilter6PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter6PFSoftPFMuonBJetTags = akCsFilter6PFbTagger.SoftPFMuonBJetTags
akCsFilter6PFSoftPFMuonByIP3dBJetTags = akCsFilter6PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter6PFSoftPFMuonByPtBJetTags = akCsFilter6PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter6PFNegativeSoftPFMuonByPtBJetTags = akCsFilter6PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter6PFPositiveSoftPFMuonByPtBJetTags = akCsFilter6PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter6PFPatJetFlavourIdLegacy = cms.Sequence(akCsFilter6PFPatJetPartonAssociationLegacy*akCsFilter6PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter6PFPatJetFlavourAssociation = akCsFilter6PFbTagger.PatJetFlavourAssociation
#akCsFilter6PFPatJetFlavourId = cms.Sequence(akCsFilter6PFPatJetPartons*akCsFilter6PFPatJetFlavourAssociation)

akCsFilter6PFJetBtaggingIP       = cms.Sequence(akCsFilter6PFImpactParameterTagInfos *
            (akCsFilter6PFTrackCountingHighEffBJetTags +
             akCsFilter6PFTrackCountingHighPurBJetTags +
             akCsFilter6PFJetProbabilityBJetTags +
             akCsFilter6PFJetBProbabilityBJetTags 
            )
            )

akCsFilter6PFJetBtaggingSV = cms.Sequence(akCsFilter6PFImpactParameterTagInfos
            *
            akCsFilter6PFSecondaryVertexTagInfos
            * (akCsFilter6PFSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter6PFSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter6PFCombinedSecondaryVertexBJetTags+
                akCsFilter6PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter6PFJetBtaggingNegSV = cms.Sequence(akCsFilter6PFImpactParameterTagInfos
            *
            akCsFilter6PFSecondaryVertexNegativeTagInfos
            * (akCsFilter6PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter6PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter6PFNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter6PFPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter6PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter6PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter6PFJetBtaggingMu = cms.Sequence(akCsFilter6PFSoftPFMuonsTagInfos * (akCsFilter6PFSoftPFMuonBJetTags
                +
                akCsFilter6PFSoftPFMuonByIP3dBJetTags
                +
                akCsFilter6PFSoftPFMuonByPtBJetTags
                +
                akCsFilter6PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter6PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter6PFJetBtagging = cms.Sequence(akCsFilter6PFJetBtaggingIP
            *akCsFilter6PFJetBtaggingSV
            *akCsFilter6PFJetBtaggingNegSV
#            *akCsFilter6PFJetBtaggingMu
            )

akCsFilter6PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter6PFJets"),
        genJetMatch          = cms.InputTag("akCsFilter6PFmatch"),
        genPartonMatch       = cms.InputTag("akCsFilter6PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter6PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter6PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter6PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter6PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter6PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter6PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter6PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter6PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter6PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter6PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter6PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter6PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter6PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter6PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter6PFJetID"),
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

akCsFilter6PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter6PFJets"),
           	    R0  = cms.double( 0.6)
)
akCsFilter6PFpatJetsWithBtagging.userData.userFloats.src += ['akCsFilter6PFNjettiness:tau1','akCsFilter6PFNjettiness:tau2','akCsFilter6PFNjettiness:tau3']

akCsFilter6PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter6PFpatJetsWithBtagging"),
                                                             genjetTag = 'ak6HiGenJets',
                                                             rParam = 0.6,
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
                                                             bTagJetName = cms.untracked.string("akCsFilter6PF"),
                                                             jetName = cms.untracked.string("akCsFilter6PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter6PFJetSequence_mc = cms.Sequence(
                                                  #akCsFilter6PFclean
                                                  #*
                                                  akCsFilter6PFmatch
                                                  *
                                                  akCsFilter6PFparton
                                                  *
                                                  akCsFilter6PFcorr
                                                  *
                                                  #akCsFilter6PFJetID
                                                  #*
                                                  akCsFilter6PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter6PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter6PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter6PFJetBtagging
                                                  *
                                                  akCsFilter6PFNjettiness
                                                  *
                                                  akCsFilter6PFpatJetsWithBtagging
                                                  *
                                                  akCsFilter6PFJetAnalyzer
                                                  )

akCsFilter6PFJetSequence_data = cms.Sequence(akCsFilter6PFcorr
                                                    *
                                                    #akCsFilter6PFJetID
                                                    #*
                                                    akCsFilter6PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter6PFJetBtagging
                                                    *
                                                    akCsFilter6PFNjettiness 
                                                    *
                                                    akCsFilter6PFpatJetsWithBtagging
                                                    *
                                                    akCsFilter6PFJetAnalyzer
                                                    )

akCsFilter6PFJetSequence_jec = cms.Sequence(akCsFilter6PFJetSequence_mc)
akCsFilter6PFJetSequence_mb = cms.Sequence(akCsFilter6PFJetSequence_mc)

akCsFilter6PFJetSequence = cms.Sequence(akCsFilter6PFJetSequence_mb)
