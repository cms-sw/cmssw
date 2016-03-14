

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter3PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter3PFJets"),
    matched = cms.InputTag("ak3HiSignalGenJets"),
    maxDeltaR = 0.3
    )

akCsFilter3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter3PFJets")
                                                        )

akCsFilter3PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter3PFJets"),
    payload = "AK3PF_offline"
    )

akCsFilter3PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter3CaloJets'))

#akCsFilter3PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak3HiSignalGenJets'))

akCsFilter3PFbTagger = bTaggers("akCsFilter3PF",0.3)

#create objects locally since they dont load properly otherwise
#akCsFilter3PFmatch = akCsFilter3PFbTagger.match
akCsFilter3PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter3PFJets"), matched = cms.InputTag("hiSignalGenParticles"))
akCsFilter3PFPatJetFlavourAssociationLegacy = akCsFilter3PFbTagger.PatJetFlavourAssociationLegacy
akCsFilter3PFPatJetPartons = akCsFilter3PFbTagger.PatJetPartons
akCsFilter3PFJetTracksAssociatorAtVertex = akCsFilter3PFbTagger.JetTracksAssociatorAtVertex
akCsFilter3PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter3PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter3PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter3PFCombinedSecondaryVertexBJetTags = akCsFilter3PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter3PFCombinedSecondaryVertexV2BJetTags = akCsFilter3PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter3PFJetBProbabilityBJetTags = akCsFilter3PFbTagger.JetBProbabilityBJetTags
akCsFilter3PFSoftPFMuonByPtBJetTags = akCsFilter3PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter3PFSoftPFMuonByIP3dBJetTags = akCsFilter3PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter3PFTrackCountingHighEffBJetTags = akCsFilter3PFbTagger.TrackCountingHighEffBJetTags
akCsFilter3PFTrackCountingHighPurBJetTags = akCsFilter3PFbTagger.TrackCountingHighPurBJetTags
akCsFilter3PFPatJetPartonAssociationLegacy = akCsFilter3PFbTagger.PatJetPartonAssociationLegacy

akCsFilter3PFImpactParameterTagInfos = akCsFilter3PFbTagger.ImpactParameterTagInfos
akCsFilter3PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter3PFJetProbabilityBJetTags = akCsFilter3PFbTagger.JetProbabilityBJetTags

akCsFilter3PFSecondaryVertexTagInfos = akCsFilter3PFbTagger.SecondaryVertexTagInfos
akCsFilter3PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter3PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter3PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter3PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter3PFCombinedSecondaryVertexBJetTags = akCsFilter3PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter3PFCombinedSecondaryVertexV2BJetTags = akCsFilter3PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter3PFSecondaryVertexNegativeTagInfos = akCsFilter3PFbTagger.SecondaryVertexNegativeTagInfos
akCsFilter3PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter3PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter3PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter3PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter3PFNegativeCombinedSecondaryVertexBJetTags = akCsFilter3PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter3PFPositiveCombinedSecondaryVertexBJetTags = akCsFilter3PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter3PFNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter3PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter3PFPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter3PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter3PFSoftPFMuonsTagInfos = akCsFilter3PFbTagger.SoftPFMuonsTagInfos
akCsFilter3PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter3PFSoftPFMuonBJetTags = akCsFilter3PFbTagger.SoftPFMuonBJetTags
akCsFilter3PFSoftPFMuonByIP3dBJetTags = akCsFilter3PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter3PFSoftPFMuonByPtBJetTags = akCsFilter3PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter3PFNegativeSoftPFMuonByPtBJetTags = akCsFilter3PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter3PFPositiveSoftPFMuonByPtBJetTags = akCsFilter3PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter3PFPatJetFlavourIdLegacy = cms.Sequence(akCsFilter3PFPatJetPartonAssociationLegacy*akCsFilter3PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter3PFPatJetFlavourAssociation = akCsFilter3PFbTagger.PatJetFlavourAssociation
#akCsFilter3PFPatJetFlavourId = cms.Sequence(akCsFilter3PFPatJetPartons*akCsFilter3PFPatJetFlavourAssociation)

akCsFilter3PFJetBtaggingIP       = cms.Sequence(akCsFilter3PFImpactParameterTagInfos *
            (akCsFilter3PFTrackCountingHighEffBJetTags +
             akCsFilter3PFTrackCountingHighPurBJetTags +
             akCsFilter3PFJetProbabilityBJetTags +
             akCsFilter3PFJetBProbabilityBJetTags 
            )
            )

akCsFilter3PFJetBtaggingSV = cms.Sequence(akCsFilter3PFImpactParameterTagInfos
            *
            akCsFilter3PFSecondaryVertexTagInfos
            * (akCsFilter3PFSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter3PFSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter3PFCombinedSecondaryVertexBJetTags+
                akCsFilter3PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter3PFJetBtaggingNegSV = cms.Sequence(akCsFilter3PFImpactParameterTagInfos
            *
            akCsFilter3PFSecondaryVertexNegativeTagInfos
            * (akCsFilter3PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter3PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter3PFNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter3PFPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter3PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter3PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter3PFJetBtaggingMu = cms.Sequence(akCsFilter3PFSoftPFMuonsTagInfos * (akCsFilter3PFSoftPFMuonBJetTags
                +
                akCsFilter3PFSoftPFMuonByIP3dBJetTags
                +
                akCsFilter3PFSoftPFMuonByPtBJetTags
                +
                akCsFilter3PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter3PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter3PFJetBtagging = cms.Sequence(akCsFilter3PFJetBtaggingIP
            *akCsFilter3PFJetBtaggingSV
            *akCsFilter3PFJetBtaggingNegSV
#            *akCsFilter3PFJetBtaggingMu
            )

akCsFilter3PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter3PFJets"),
        genJetMatch          = cms.InputTag("akCsFilter3PFmatch"),
        genPartonMatch       = cms.InputTag("akCsFilter3PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter3PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter3PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter3PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter3PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter3PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter3PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter3PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter3PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter3PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter3PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter3PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter3PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter3PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter3PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter3PFJetID"),
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

akCsFilter3PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter3PFJets"),
           	    R0  = cms.double( 0.3)
)
akCsFilter3PFpatJetsWithBtagging.userData.userFloats.src += ['akCsFilter3PFNjettiness:tau1','akCsFilter3PFNjettiness:tau2','akCsFilter3PFNjettiness:tau3']

akCsFilter3PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter3PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsFilter3PF"),
                                                             jetName = cms.untracked.string("akCsFilter3PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter3PFJetSequence_mc = cms.Sequence(
                                                  #akCsFilter3PFclean
                                                  #*
                                                  akCsFilter3PFmatch
                                                  *
                                                  akCsFilter3PFparton
                                                  *
                                                  akCsFilter3PFcorr
                                                  *
                                                  #akCsFilter3PFJetID
                                                  #*
                                                  akCsFilter3PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter3PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter3PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter3PFJetBtagging
                                                  *
                                                  akCsFilter3PFNjettiness
                                                  *
                                                  akCsFilter3PFpatJetsWithBtagging
                                                  *
                                                  akCsFilter3PFJetAnalyzer
                                                  )

akCsFilter3PFJetSequence_data = cms.Sequence(akCsFilter3PFcorr
                                                    *
                                                    #akCsFilter3PFJetID
                                                    #*
                                                    akCsFilter3PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter3PFJetBtagging
                                                    *
                                                    akCsFilter3PFNjettiness 
                                                    *
                                                    akCsFilter3PFpatJetsWithBtagging
                                                    *
                                                    akCsFilter3PFJetAnalyzer
                                                    )

akCsFilter3PFJetSequence_jec = cms.Sequence(akCsFilter3PFJetSequence_mc)
akCsFilter3PFJetSequence_mb = cms.Sequence(akCsFilter3PFJetSequence_mc)

akCsFilter3PFJetSequence = cms.Sequence(akCsFilter3PFJetSequence_jec)
akCsFilter3PFJetAnalyzer.genPtMin = cms.untracked.double(1)
akCsFilter3PFJetAnalyzer.jetPtMin = cms.double(1)
