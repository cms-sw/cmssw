

import FWCore.ParameterSet.Config as cms
from HeavyIonsAnalysis.JetAnalysis.patHeavyIonSequences_cff import patJetGenJetMatch, patJetPartonMatch, patJetCorrFactors, patJets
from HeavyIonsAnalysis.JetAnalysis.inclusiveJetAnalyzer_cff import *
from HeavyIonsAnalysis.JetAnalysis.bTaggers_cff import *
from RecoJets.JetProducers.JetIDParams_cfi import *
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness

akCsFilter5PFmatch = patJetGenJetMatch.clone(
    src = cms.InputTag("akCsFilter5PFJets"),
    matched = cms.InputTag("ak5HiSignalGenJets"),
    maxDeltaR = 0.5
    )

akCsFilter5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter5PFJets")
                                                        )

akCsFilter5PFcorr = patJetCorrFactors.clone(
    useNPV = cms.bool(False),
    useRho = cms.bool(False),
#    primaryVertices = cms.InputTag("hiSelectedVertex"),
    levels   = cms.vstring('L2Relative','L3Absolute'),
    src = cms.InputTag("akCsFilter5PFJets"),
    payload = "AK5PF_offline"
    )

akCsFilter5PFJetID= cms.EDProducer('JetIDProducer', JetIDParams, src = cms.InputTag('akCsFilter5CaloJets'))

#akCsFilter5PFclean   = heavyIonCleanedGenJets.clone(src = cms.InputTag('ak5HiSignalGenJets'))

akCsFilter5PFbTagger = bTaggers("akCsFilter5PF",0.5)

#create objects locally since they dont load properly otherwise
#akCsFilter5PFmatch = akCsFilter5PFbTagger.match
akCsFilter5PFparton = patJetPartonMatch.clone(src = cms.InputTag("akCsFilter5PFJets"), matched = cms.InputTag("hiSignalGenParticles"))
akCsFilter5PFPatJetFlavourAssociationLegacy = akCsFilter5PFbTagger.PatJetFlavourAssociationLegacy
akCsFilter5PFPatJetPartons = akCsFilter5PFbTagger.PatJetPartons
akCsFilter5PFJetTracksAssociatorAtVertex = akCsFilter5PFbTagger.JetTracksAssociatorAtVertex
akCsFilter5PFJetTracksAssociatorAtVertex.tracks = cms.InputTag("highPurityTracks")
akCsFilter5PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter5PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter5PFCombinedSecondaryVertexBJetTags = akCsFilter5PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter5PFCombinedSecondaryVertexV2BJetTags = akCsFilter5PFbTagger.CombinedSecondaryVertexV2BJetTags
akCsFilter5PFJetBProbabilityBJetTags = akCsFilter5PFbTagger.JetBProbabilityBJetTags
akCsFilter5PFSoftPFMuonByPtBJetTags = akCsFilter5PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter5PFSoftPFMuonByIP3dBJetTags = akCsFilter5PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter5PFTrackCountingHighEffBJetTags = akCsFilter5PFbTagger.TrackCountingHighEffBJetTags
akCsFilter5PFTrackCountingHighPurBJetTags = akCsFilter5PFbTagger.TrackCountingHighPurBJetTags
akCsFilter5PFPatJetPartonAssociationLegacy = akCsFilter5PFbTagger.PatJetPartonAssociationLegacy

akCsFilter5PFImpactParameterTagInfos = akCsFilter5PFbTagger.ImpactParameterTagInfos
akCsFilter5PFImpactParameterTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter5PFJetProbabilityBJetTags = akCsFilter5PFbTagger.JetProbabilityBJetTags

akCsFilter5PFSecondaryVertexTagInfos = akCsFilter5PFbTagger.SecondaryVertexTagInfos
akCsFilter5PFSimpleSecondaryVertexHighEffBJetTags = akCsFilter5PFbTagger.SimpleSecondaryVertexHighEffBJetTags
akCsFilter5PFSimpleSecondaryVertexHighPurBJetTags = akCsFilter5PFbTagger.SimpleSecondaryVertexHighPurBJetTags
akCsFilter5PFCombinedSecondaryVertexBJetTags = akCsFilter5PFbTagger.CombinedSecondaryVertexBJetTags
akCsFilter5PFCombinedSecondaryVertexV2BJetTags = akCsFilter5PFbTagger.CombinedSecondaryVertexV2BJetTags

akCsFilter5PFSecondaryVertexNegativeTagInfos = akCsFilter5PFbTagger.SecondaryVertexNegativeTagInfos
akCsFilter5PFNegativeSimpleSecondaryVertexHighEffBJetTags = akCsFilter5PFbTagger.NegativeSimpleSecondaryVertexHighEffBJetTags
akCsFilter5PFNegativeSimpleSecondaryVertexHighPurBJetTags = akCsFilter5PFbTagger.NegativeSimpleSecondaryVertexHighPurBJetTags
akCsFilter5PFNegativeCombinedSecondaryVertexBJetTags = akCsFilter5PFbTagger.NegativeCombinedSecondaryVertexBJetTags
akCsFilter5PFPositiveCombinedSecondaryVertexBJetTags = akCsFilter5PFbTagger.PositiveCombinedSecondaryVertexBJetTags
akCsFilter5PFNegativeCombinedSecondaryVertexV2BJetTags = akCsFilter5PFbTagger.NegativeCombinedSecondaryVertexV2BJetTags
akCsFilter5PFPositiveCombinedSecondaryVertexV2BJetTags = akCsFilter5PFbTagger.PositiveCombinedSecondaryVertexV2BJetTags

akCsFilter5PFSoftPFMuonsTagInfos = akCsFilter5PFbTagger.SoftPFMuonsTagInfos
akCsFilter5PFSoftPFMuonsTagInfos.primaryVertex = cms.InputTag("offlinePrimaryVertices")
akCsFilter5PFSoftPFMuonBJetTags = akCsFilter5PFbTagger.SoftPFMuonBJetTags
akCsFilter5PFSoftPFMuonByIP3dBJetTags = akCsFilter5PFbTagger.SoftPFMuonByIP3dBJetTags
akCsFilter5PFSoftPFMuonByPtBJetTags = akCsFilter5PFbTagger.SoftPFMuonByPtBJetTags
akCsFilter5PFNegativeSoftPFMuonByPtBJetTags = akCsFilter5PFbTagger.NegativeSoftPFMuonByPtBJetTags
akCsFilter5PFPositiveSoftPFMuonByPtBJetTags = akCsFilter5PFbTagger.PositiveSoftPFMuonByPtBJetTags
akCsFilter5PFPatJetFlavourIdLegacy = cms.Sequence(akCsFilter5PFPatJetPartonAssociationLegacy*akCsFilter5PFPatJetFlavourAssociationLegacy)
#Not working with our PU sub, but keep it here for reference
#akCsFilter5PFPatJetFlavourAssociation = akCsFilter5PFbTagger.PatJetFlavourAssociation
#akCsFilter5PFPatJetFlavourId = cms.Sequence(akCsFilter5PFPatJetPartons*akCsFilter5PFPatJetFlavourAssociation)

akCsFilter5PFJetBtaggingIP       = cms.Sequence(akCsFilter5PFImpactParameterTagInfos *
            (akCsFilter5PFTrackCountingHighEffBJetTags +
             akCsFilter5PFTrackCountingHighPurBJetTags +
             akCsFilter5PFJetProbabilityBJetTags +
             akCsFilter5PFJetBProbabilityBJetTags 
            )
            )

akCsFilter5PFJetBtaggingSV = cms.Sequence(akCsFilter5PFImpactParameterTagInfos
            *
            akCsFilter5PFSecondaryVertexTagInfos
            * (akCsFilter5PFSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter5PFSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter5PFCombinedSecondaryVertexBJetTags+
                akCsFilter5PFCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter5PFJetBtaggingNegSV = cms.Sequence(akCsFilter5PFImpactParameterTagInfos
            *
            akCsFilter5PFSecondaryVertexNegativeTagInfos
            * (akCsFilter5PFNegativeSimpleSecondaryVertexHighEffBJetTags+
                akCsFilter5PFNegativeSimpleSecondaryVertexHighPurBJetTags+
                akCsFilter5PFNegativeCombinedSecondaryVertexBJetTags+
                akCsFilter5PFPositiveCombinedSecondaryVertexBJetTags+
                akCsFilter5PFNegativeCombinedSecondaryVertexV2BJetTags+
                akCsFilter5PFPositiveCombinedSecondaryVertexV2BJetTags
              )
            )

akCsFilter5PFJetBtaggingMu = cms.Sequence(akCsFilter5PFSoftPFMuonsTagInfos * (akCsFilter5PFSoftPFMuonBJetTags
                +
                akCsFilter5PFSoftPFMuonByIP3dBJetTags
                +
                akCsFilter5PFSoftPFMuonByPtBJetTags
                +
                akCsFilter5PFNegativeSoftPFMuonByPtBJetTags
                +
                akCsFilter5PFPositiveSoftPFMuonByPtBJetTags
              )
            )

akCsFilter5PFJetBtagging = cms.Sequence(akCsFilter5PFJetBtaggingIP
            *akCsFilter5PFJetBtaggingSV
            *akCsFilter5PFJetBtaggingNegSV
#            *akCsFilter5PFJetBtaggingMu
            )

akCsFilter5PFpatJetsWithBtagging = patJets.clone(jetSource = cms.InputTag("akCsFilter5PFJets"),
        genJetMatch          = cms.InputTag("akCsFilter5PFmatch"),
        genPartonMatch       = cms.InputTag("akCsFilter5PFparton"),
        jetCorrFactorsSource = cms.VInputTag(cms.InputTag("akCsFilter5PFcorr")),
        JetPartonMapSource   = cms.InputTag("akCsFilter5PFPatJetFlavourAssociationLegacy"),
	JetFlavourInfoSource   = cms.InputTag("akCsFilter5PFPatJetFlavourAssociation"),
        trackAssociationSource = cms.InputTag("akCsFilter5PFJetTracksAssociatorAtVertex"),
	useLegacyJetMCFlavour = True,
        discriminatorSources = cms.VInputTag(cms.InputTag("akCsFilter5PFSimpleSecondaryVertexHighEffBJetTags"),
            cms.InputTag("akCsFilter5PFSimpleSecondaryVertexHighPurBJetTags"),
            cms.InputTag("akCsFilter5PFCombinedSecondaryVertexBJetTags"),
            cms.InputTag("akCsFilter5PFCombinedSecondaryVertexV2BJetTags"),
            cms.InputTag("akCsFilter5PFJetBProbabilityBJetTags"),
            cms.InputTag("akCsFilter5PFJetProbabilityBJetTags"),
            #cms.InputTag("akCsFilter5PFSoftPFMuonByPtBJetTags"),
            #cms.InputTag("akCsFilter5PFSoftPFMuonByIP3dBJetTags"),
            cms.InputTag("akCsFilter5PFTrackCountingHighEffBJetTags"),
            cms.InputTag("akCsFilter5PFTrackCountingHighPurBJetTags"),
            ),
        jetIDMap = cms.InputTag("akCsFilter5PFJetID"),
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

akCsFilter5PFNjettiness = Njettiness.clone(
		    src = cms.InputTag("akCsFilter5PFJets"),
           	    R0  = cms.double( 0.5)
)
akCsFilter5PFpatJetsWithBtagging.userData.userFloats.src += ['akCsFilter5PFNjettiness:tau1','akCsFilter5PFNjettiness:tau2','akCsFilter5PFNjettiness:tau3']

akCsFilter5PFJetAnalyzer = inclusiveJetAnalyzer.clone(jetTag = cms.InputTag("akCsFilter5PFpatJetsWithBtagging"),
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
                                                             bTagJetName = cms.untracked.string("akCsFilter5PF"),
                                                             jetName = cms.untracked.string("akCsFilter5PF"),
                                                             genPtMin = cms.untracked.double(15),
                                                             hltTrgResults = cms.untracked.string('TriggerResults::'+'HISIGNAL'),
							     doTower = cms.untracked.bool(True),
							     doSubJets = cms.untracked.bool(True)
                                                             )

akCsFilter5PFJetSequence_mc = cms.Sequence(
                                                  #akCsFilter5PFclean
                                                  #*
                                                  akCsFilter5PFmatch
                                                  *
                                                  akCsFilter5PFparton
                                                  *
                                                  akCsFilter5PFcorr
                                                  *
                                                  #akCsFilter5PFJetID
                                                  #*
                                                  akCsFilter5PFPatJetFlavourIdLegacy
                                                  #*
			                          #akCsFilter5PFPatJetFlavourId  # Use legacy algo till PU implemented
                                                  *
                                                  akCsFilter5PFJetTracksAssociatorAtVertex
                                                  *
                                                  akCsFilter5PFJetBtagging
                                                  *
                                                  akCsFilter5PFNjettiness
                                                  *
                                                  akCsFilter5PFpatJetsWithBtagging
                                                  *
                                                  akCsFilter5PFJetAnalyzer
                                                  )

akCsFilter5PFJetSequence_data = cms.Sequence(akCsFilter5PFcorr
                                                    *
                                                    #akCsFilter5PFJetID
                                                    #*
                                                    akCsFilter5PFJetTracksAssociatorAtVertex
                                                    *
                                                    akCsFilter5PFJetBtagging
                                                    *
                                                    akCsFilter5PFNjettiness 
                                                    *
                                                    akCsFilter5PFpatJetsWithBtagging
                                                    *
                                                    akCsFilter5PFJetAnalyzer
                                                    )

akCsFilter5PFJetSequence_jec = cms.Sequence(akCsFilter5PFJetSequence_mc)
akCsFilter5PFJetSequence_mb = cms.Sequence(akCsFilter5PFJetSequence_mc)

akCsFilter5PFJetSequence = cms.Sequence(akCsFilter5PFJetSequence_data)
